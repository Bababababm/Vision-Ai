
/*
  VISION AI — COMPLETE APP.JS
  Mobile web app for Android / Chrome

  IMPORTANT:
  Replace the entire old app.js with this code.
  Do not append it to the existing code.
*/

const $ = id => document.getElementById(id);

const video = $('video');
const canvas = $('overlay');
const ctx = canvas ? canvas.getContext('2d') : null;

const MP_VERSION = '0.10.22';
const MP_URL =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models';

let FaceLandmarker;
let HandLandmarker;
let FilesetResolver;

let faceTask = null;
let handTask = null;
let stream = null;

let running = false;
let starting = false;
let front = true;
let animationId = null;
let lastVideoTime = -1;
let lastAnalysisError = '';

let lastFace = null;
let lastFaceCount = 0;
let previousHead = null;

let db = null;
let encryptionKey = null;
let profiles = [];

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// =====================================
// STATUS AND ERROR HANDLING
// =====================================

function setText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

function status(message) {
  setText('status', message);
  console.log('[Vision AI]', message);
}

function libraryStatus(message) {
  setText('libraryStatus', message);
}

function errorMessage(error) {
  return error?.message || String(error);
}

function showError(error) {
  const message = errorMessage(error);
  console.error('[Vision AI ERROR]', error);
  status('Error: ' + message);
  alert('Vision AI error:\n' + message);
}

function updateButtons() {
  if ($('start')) {
    $('start').disabled = starting || running;
  }

  if ($('stop')) {
    $('stop').disabled = !running && !starting;
  }

  if ($('switch')) {
    $('switch').disabled = starting;
  }

  if ($('unlock')) {
    $('unlock').disabled = !!encryptionKey;
  }

  if ($('lock')) {
    $('lock').disabled = !encryptionKey;
  }

  if ($('save')) {
    $('save').disabled =
      !running ||
      !encryptionKey ||
      !lastFace ||
      lastFaceCount !== 1;
  }
}

// =====================================
// CAMERA
// =====================================

async function startCamera() {
  if (starting || running) return;

  starting = true;
  updateButtons();

  try {
    status('Start button pressed');

    if (!video || !canvas || !ctx) {
      throw new Error(
        'Video or overlay is missing from index.html'
      );
    }

    if (!window.isSecureContext) {
      throw new Error(
        'Open Vision AI using its HTTPS GitHub Pages URL.'
      );
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        'Camera API unavailable. Open the site directly in Chrome.'
      );
    }

    status('Requesting camera permission...');

    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: {
          ideal: front ? 'user' : 'environment'
        },
        width: { ideal: 640 },
        height: { ideal: 480 }
      }
    });

    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;

    await video.play();

    if ($('placeholder')) {
      $('placeholder').hidden = true;
    }

    running = true;
    starting = false;
    lastVideoTime = -1;

    updateButtons();
    status('Camera active — loading AI models...');

    await initializeModels();

    if (!running) return;

    status('Analyzing locally');
    analyzeFrame();

  } catch (error) {
    running = false;
    starting = false;
    cleanupCamera();
    updateButtons();
    showError(error);
  }
}

function cleanupCamera() {
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }

  stream = null;

  if (video) {
    video.pause();
    video.srcObject = null;
  }

  if (ctx && canvas) {
    ctx.clearRect(
      0, 0, canvas.width, canvas.height
    );
  }

  if ($('placeholder')) {
    $('placeholder').hidden = false;
  }

  lastFace = null;
  lastFaceCount = 0;
  previousHead = null;
  lastVideoTime = -1;
}

function stopCamera() {
  running = false;
  starting = false;
  cleanupCamera();
  updateButtons();
  status('Stopped');
}

async function switchCamera() {
  if (starting) return;

  const restart = running;
  stopCamera();
  front = !front;

  if (restart) {
    await startCamera();
  } else {
    status(
      front
        ? 'Front camera selected'
        : 'Rear camera selected'
    );
  }
}

// =====================================
// MEDIAPIPE MODELS
// =====================================

async function initializeModels() {
  if (faceTask && handTask) return;

  if (!FilesetResolver) {
    status('Downloading MediaPipe library...');

    const mp = await import(`${MP_URL}/+esm`);

    FaceLandmarker = mp.FaceLandmarker;
    HandLandmarker = mp.HandLandmarker;
    FilesetResolver = mp.FilesetResolver;

    if (
      !FaceLandmarker ||
      !HandLandmarker ||
      !FilesetResolver
    ) {
      throw new Error(
        'MediaPipe library exports are missing.'
      );
    }
  }

  status('Loading AI runtime...');

  const vision =
    await FilesetResolver.forVisionTasks(
      `${MP_URL}/wasm`
    );

  if (!faceTask) {
    status('Loading face model...');

    faceTask = await FaceLandmarker.createFromOptions(
      vision,
      {
        baseOptions: {
          modelAssetPath:
            `${MODEL_URL}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          delegate: 'CPU'
        },
        runningMode: 'VIDEO',
        numFaces: 2,
        outputFaceBlendshapes: true
      }
    );
  }

  if (!handTask) {
    status('Loading hand model...');

    handTask = await HandLandmarker.createFromOptions(
      vision,
      {
        baseOptions: {
          modelAssetPath:
            `${MODEL_URL}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
          delegate: 'CPU'
        },
        runningMode: 'VIDEO',
        numHands: 2
      }
    );
  }

  status('AI models ready');
}

// =====================================
// FACE GEOMETRY
// =====================================

function distance(a, b) {
  return Math.hypot(
    a.x - b.x,
    a.y - b.y
  );
}

function ratio(a, b, c, d) {
  return distance(a, b) /
    Math.max(0.00001, distance(c, d));
}

// Experimental face geometry profile.
// NOT reliable identity recognition.
function createFaceTemplate(landmarks) {
  const pairs = [
    [33, 263],
    [61, 291],
    [1, 152],
    [4, 168],
    [133, 362],
    [70, 300],
    [234, 454],
    [10, 152],
    [13, 14],
    [159, 145],
    [386, 374],
    [105, 334],
    [58, 288],
    [172, 397]
  ];

  const scale =
    distance(landmarks[33], landmarks[263]) || 1;

  return pairs.map(([a, b]) =>
    Math.min(
      5,
      distance(
        landmarks[a],
        landmarks[b]
      ) / scale
    )
  );
}

function compareTemplates(a, b) {
  if (
    !Array.isArray(a) ||
    !Array.isArray(b) ||
    !a.length ||
    a.length !== b.length
  ) {
    return Infinity;
  }

  const sum = a.reduce(
    (total, value, i) =>
      total + (value - b[i]) ** 2,
    0
  );

  return Math.sqrt(sum / a.length);
}

function findClosestProfile(template) {
  if (!encryptionKey || !profiles.length) {
    return null;
  }

  let closest = null;

  for (const profile of profiles) {
    const score = compareTemplates(
      template,
      profile.template
    );

    if (
      Number.isFinite(score) &&
      (!closest || score < closest.score)
    ) {
      closest = {
        name: profile.name,
        score
      };
    }
  }

  return closest;
}

// =====================================
// HAND GESTURES
// =====================================

function detectGesture(landmarks) {
  const wrist = landmarks[0];

  const tips = [8, 12, 16, 20];
  const joints = [6, 10, 14, 18];

  const extended = tips.map((tip, i) =>
    distance(landmarks[tip], wrist) >
    distance(landmarks[joints[i]], wrist) * 1.13
  );

  const count = extended.filter(Boolean).length;

  const pinch =
    distance(landmarks[4], landmarks[8]) <
    distance(landmarks[5], landmarks[17]) * 0.25;

  if (pinch) return 'Pinch';
  if (count === 4) return 'Open palm';
  if (count === 0) return 'Fist';

  if (extended[0] && count === 1) {
    return 'Pointing';
  }

  if (
    extended[0] &&
    extended[1] &&
    count === 2
  ) {
    return 'Peace sign';
  }

  return `${count} fingers extended`;
}

// =====================================
// HEAD, EYES AND MOUTH
// =====================================

function analyzeFace(landmarks, blendshapes) {
  const eyeWidth =
    distance(landmarks[33], landmarks[263]) || 1;

  const offset =
    (
      landmarks[1].x -
      (landmarks[33].x + landmarks[263].x) / 2
    ) / eyeWidth;

  const headDirection =
    offset > 0.10
      ? 'Right'
      : offset < -0.10
        ? 'Left'
        : 'Center';

  const leftEye = ratio(
    landmarks[159],
    landmarks[145],
    landmarks[33],
    landmarks[133]
  );

  const rightEye = ratio(
    landmarks[386],
    landmarks[374],
    landmarks[362],
    landmarks[263]
  );

  const mouthRatio = ratio(
    landmarks[13],
    landmarks[14],
    landmarks[61],
    landmarks[291]
  );

  const scores = Object.fromEntries(
    (blendshapes || []).map(item => [
      item.categoryName,
      item.score
    ])
  );

  const blinking =
    (
      (scores.eyeBlinkLeft ?? 0) > 0.55 &&
      (scores.eyeBlinkRight ?? 0) > 0.55
    ) ||
    (leftEye < 0.12 && rightEye < 0.12);

  const mouthOpen = mouthRatio > 0.10;

  const centerY =
    (landmarks[1].y + landmarks[152].y) / 2;

  const now = performance.now();
  let movement = '';

  if (
    previousHead &&
    now - previousHead.time < 1000
  ) {
    const change = centerY - previousHead.y;

    if (Math.abs(change) > 0.025) {
      movement = change > 0
        ? ' · Moving down'
        : ' · Moving up';
    }
  }

  previousHead = {
    y: centerY,
    time: now
  };

  return (
    `Head: ${headDirection} · ` +
    `Eyes: ${blinking ? 'Blink/closed' : 'Open'} · ` +
    `Mouth: ${mouthOpen ? 'Open' : 'Closed'}` +
    movement
  );
}

// =====================================
// DRAW LANDMARKS
// =====================================

function drawLandmarks(landmarks, color) {
  if (!ctx) return;

  ctx.fillStyle = color;

  for (const point of landmarks) {
    const x =
      (front ? 1 - point.x : point.x) *
      canvas.width;

    const y = point.y * canvas.height;

    ctx.beginPath();
    ctx.arc(x, y, 1.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// =====================================
// LIVE ANALYSIS
// =====================================

function analyzeFrame() {
  if (!running) return;

  try {
    if (
      faceTask &&
      handTask &&
      video.readyState >= 2 &&
      video.currentTime !== lastVideoTime
    ) {
      lastVideoTime = video.currentTime;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const now = performance.now();

      const faceResults =
        faceTask.detectForVideo(video, now);

      const handResults =
        handTask.detectForVideo(video, now);

      ctx.clearRect(
        0, 0, canvas.width, canvas.height
      );

      const faces =
        faceResults.faceLandmarks || [];

      const hands =
        handResults.landmarks || [];

      lastFaceCount = faces.length;

      lastFace = faces.length === 1
        ? createFaceTemplate(faces[0])
        : null;

      faces.forEach(face =>
        drawLandmarks(face, '#5df4bd')
      );

      hands.forEach(hand =>
        drawLandmarks(hand, '#ffc970')
      );

      setText(
        'faceInfo',
        `Faces: ${faces.length}`
      );

      setText(
        'handInfo',
        `Hands: ${hands.length}` +
        (
          hands.length
            ? ' · ' + hands.map(detectGesture).join(' / ')
            : ''
        )
      );

      if (faces.length === 1) {
        const blendshapes =
          faceResults.faceBlendshapes?.[0]?.categories;

        setText(
          'motionInfo',
          analyzeFace(faces[0], blendshapes)
        );

        const match =
          findClosestProfile(lastFace);

        if (match) {
          setText(
            'matchInfo',
            `Closest saved profile: ${match.name} ` +
            `(distance ${match.score.toFixed(3)}; ` +
            'identity unverified)'
          );
        } else {
          setText(
            'matchInfo',
            encryptionKey
              ? 'No saved profiles'
              : 'Face library locked'
          );
        }

      } else {
        setText('motionInfo', 'Movement: —');

        setText(
          'matchInfo',
          faces.length > 1
            ? 'Multiple faces — matching paused'
            : 'Saved face: —'
        );

        previousHead = null;
      }

      updateButtons();
    }

  } catch (error) {
    const message = errorMessage(error);

    if (message !== lastAnalysisError) {
      lastAnalysisError = message;
      console.error(error);
      status('Analysis error: ' + message);
    }
  }

  animationId = requestAnimationFrame(analyzeFrame);
}

// =====================================
// INDEXEDDB
// =====================================

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(
      'vision-ai-private-v1',
      1
    );

    request.onupgradeneeded = () => {
      const database = request.result;

      if (!database.objectStoreNames.contains('data')) {
        database.createObjectStore('data');
      }
    };

    request.onsuccess = () =>
      resolve(request.result);

    request.onerror = () =>
      reject(request.error);
  });
}

function dbGet(name) {
  return new Promise((resolve, reject) => {
    const request = db
      .transaction('data')
      .objectStore('data')
      .get(name);

    request.onsuccess = () =>
      resolve(request.result);

    request.onerror = () =>
      reject(request.error);
  });
}

function dbPut(name, value) {
  return new Promise((resolve, reject) => {
    const request = db
      .transaction('data', 'readwrite')
      .objectStore('data')
      .put(value, name);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// =====================================
// ENCRYPTION
// =====================================

function toBase64(value) {
  return btoa(
    String.fromCharCode(
      ...new Uint8Array(value)
    )
  );
}

function fromBase64(value) {
  return Uint8Array.from(
    atob(value),
    char => char.charCodeAt(0)
  );
}

async function deriveKey(passphrase, salt) {
  const material =
    await crypto.subtle.importKey(
      'raw',
      encoder.encode(passphrase),
      'PBKDF2',
      false,
      ['deriveKey']
    );

  return crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt,
      iterations: 250000,
      hash: 'SHA-256'
    },
    material,
    {
      name: 'AES-GCM',
      length: 256
    },
    false,
    ['encrypt', 'decrypt']
  );
}

async function encryptProfiles(data) {
  const iv = crypto.getRandomValues(
    new Uint8Array(12)
  );

  const encrypted =
    await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv },
      encryptionKey,
      encoder.encode(JSON.stringify(data))
    );

  return {
    iv: toBase64(iv),
    cipher: toBase64(encrypted)
  };
}

async function saveLibrary() {
  if (!db || !encryptionKey) {
    throw new Error('Face library is locked.');
  }

  await dbPut(
    'vault',
    await encryptProfiles(profiles)
  );

  renderProfiles();
}

// =====================================
// FACE LIBRARY
// =====================================

function renderProfiles() {
  const list = $('people');
  if (!list) return;

  list.replaceChildren();

  for (const profile of profiles) {
    const item = document.createElement('li');
    const name = document.createElement('span');
    const button = document.createElement('button');

    name.textContent = profile.name;
    button.textContent = 'Delete';

    button.onclick = async () => {
      if (!confirm(`Delete ${profile.name}?`)) return;

      const previous = profiles;

      try {
        profiles = profiles.filter(
          p => p.id !== profile.id
        );

        await saveLibrary();
        libraryStatus('Profile deleted');

      } catch (error) {
        profiles = previous;
        renderProfiles();
        libraryStatus(
          'Delete failed: ' + errorMessage(error)
        );
      }
    };

    item.append(name, button);
    list.append(item);
  }
}

async function unlockLibrary() {
  try {
    const passphrase = $('pass')?.value || '';

    if (passphrase.length < 10) {
      throw new Error(
        'Passphrase must contain at least 10 characters.'
      );
    }

    db ??= await openDatabase();

    const storedSalt = await dbGet('salt');
    const existing = !!storedSalt;

    const salt = existing
      ? fromBase64(storedSalt)
      : crypto.getRandomValues(new Uint8Array(16));

    const candidateKey =
      await deriveKey(passphrase, salt);

    const vault = await dbGet('vault');
    let loadedProfiles = [];

    if (vault) {
      const decrypted =
        await crypto.subtle.decrypt(
          {
            name: 'AES-GCM',
            iv: fromBase64(vault.iv)
          },
          candidateKey,
          fromBase64(vault.cipher)
        );

      loadedProfiles = JSON.parse(
        decoder.decode(decrypted)
      );

      if (!Array.isArray(loadedProfiles)) {
        throw new Error('Invalid library data.');
      }

    } else if (existing) {
      throw new Error(
        'Encrypted library data is missing.'
      );
    }

    encryptionKey = candidateKey;
    profiles = loadedProfiles;

    if (!existing) {
      await saveLibrary();
      await dbPut('salt', toBase64(salt));
    }

    $('pass').value = '';

    renderProfiles();
    updateButtons();

    libraryStatus(
      `Unlocked · ${profiles.length} saved profiles`
    );

  } catch (error) {
    encryptionKey = null;
    profiles = [];
    renderProfiles();
    updateButtons();

    libraryStatus(
      'Unlock failed: ' + errorMessage(error)
    );
  }
}

function lockLibrary() {
  encryptionKey = null;
  profiles = [];

  renderProfiles();
  updateButtons();
  libraryStatus('Library locked');
}

async function saveFace() {
  try {
    if (
      !encryptionKey ||
      !lastFace ||
      lastFaceCount !== 1
    ) {
      throw new Error(
        'Unlock the library and show exactly one face.'
      );
    }

    if (!$('consent')?.checked) {
      throw new Error(
        'Confirm that the person consented to enrollment.'
      );
    }

    const name = $('person')?.value.trim() || '';

    if (!name || name.length > 80) {
      throw new Error(
        'Enter a name between 1 and 80 characters.'
      );
    }

    const profile = {
      id: crypto.randomUUID(),
      name,
      template: [...lastFace],
      createdAt: new Date().toISOString()
    };

    profiles.push(profile);

    try {
      await saveLibrary();
    } catch (error) {
      profiles = profiles.filter(
        p => p.id !== profile.id
      );
      throw error;
    }

    $('person').value = '';
    $('consent').checked = false;

    libraryStatus(
      `${name} saved in encrypted local library`
    );

  } catch (error) {
    libraryStatus(errorMessage(error));
  }
}

// =====================================
// BUTTON INITIALIZATION
// =====================================

function initializeApp() {
  console.log('Vision AI initializing...');

  const startButton = $('start');

  if (!startButton) {
    console.error('Start button missing from HTML');
    status('Error: Start button missing');
    return;
  }

  // Connect camera button first.
  startButton.onclick = () => {
    status('Start button pressed');
    startCamera();
  };

  if ($('stop')) {
    $('stop').onclick = stopCamera;
  }

  if ($('switch')) {
    $('switch').onclick = switchCamera;
  }

  if ($('unlock')) {
    $('unlock').onclick = unlockLibrary;
  }

  if ($('lock')) {
    $('lock').onclick = lockLibrary;
  }

  if ($('save')) {
    $('save').onclick = saveFace;
  }

  updateButtons();

  status('Vision AI ready — tap Start camera');
  libraryStatus('Library locked');

  console.log('Vision AI initialized successfully');
}

// =====================================
// GLOBAL ERROR REPORTING
// =====================================

window.addEventListener('error', event => {
  console.error('JavaScript error:', event.message);
  status('JavaScript error: ' + event.message);
});

window.addEventListener('unhandledrejection', event => {
  const message = errorMessage(event.reason);
  console.error('Unhandled error:', message);
  status('Error: ' + message);
});

window.addEventListener('pagehide', () => {
  running = false;
  starting = false;
  cleanupCamera();
  encryptionKey = null;
  profiles = [];
});

// =====================================
// START APP
// =====================================

if (document.readyState === 'loading') {
  document.addEventListener(
    'DOMContentLoaded',
    initializeApp,
    { once: true }
  );
} else {
  initializeApp();
}
