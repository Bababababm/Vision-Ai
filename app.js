
/*
 VISION AI — MOBILE WEB APP
 Complete replacement for app.js

 Features:
 - Camera startup and switching
 - Face landmark tracking
 - Hand gesture estimates
 - Head, eye and mouth movement
 - Experimental saved-face comparison
 - Encrypted local face library
 - Alternative MediaPipe CDN loading
 - Visible error diagnostics
*/

const $ = id => document.getElementById(id);

const video = $('video');
const canvas = $('overlay');
const ctx = canvas?.getContext('2d');

const VERSION = '1.2.2';

const MODEL_BASE =
  'https://storage.googleapis.com/mediapipe-models';

const CDN_SOURCES = [
  {
    name: 'esm.sh',
    module:
      `https://esm.sh/@mediapipe/tasks-vision@${VERSION}`,
    wasm:
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`
  },
  {
    name: 'jsDelivr',
    module:
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/+esm`,
    wasm:
      `https://unpkg.com/@mediapipe/tasks-vision@${VERSION}/wasm`
  }
];

let FaceLandmarker;
let HandLandmarker;
let FilesetResolver;

let faceTask = null;
let handTask = null;
let stream = null;

let running = false;
let starting = false;
let front = true;
let raf = null;
let session = 0;
let lastVideoTime = -1;
let lastError = '';

let lastFace = null;
let lastFaceCount = 0;
let previousHead = null;

let db = null;
let encryptionKey = null;
let profiles = [];

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// ====================================
// STATUS AND DIAGNOSTICS
// ====================================

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function status(message) {
  console.log('[Vision AI]', message);
  setText('status', message);
}

function libraryStatus(message) {
  setText('libraryStatus', message);
}

function errorMessage(error) {
  return error?.message || String(error);
}

function showError(error) {
  const message = errorMessage(error);

  console.error('[Vision AI]', error);
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

// ====================================
// MEDIAPIPE LOADER
// ====================================

async function loadMediaPipe() {
  if (FilesetResolver) return;

  let errors = [];

  for (const source of CDN_SOURCES) {
    try {
      status('Loading AI from ' + source.name);

      const mp = await import(
        /* @vite-ignore */
        source.module
      );

      const resolver = mp.FilesetResolver;
      const face = mp.FaceLandmarker;
      const hand = mp.HandLandmarker;

      if (!resolver || !face || !hand) {
        throw new Error(
          'Required MediaPipe exports missing'
        );
      }

      FilesetResolver = resolver;
      FaceLandmarker = face;
      HandLandmarker = hand;

      status('AI library loaded');
      return;

    } catch (error) {
      console.error(
        source.name + ' failed:',
        error
      );

      errors.push(
        source.name + ': ' + errorMessage(error)
      );
    }
  }

  throw new Error(
    'Unable to download MediaPipe library. ' +
    errors.join(' | ')
  );
}

async function initializeModels() {
  if (faceTask && handTask) return;

  await loadMediaPipe();

  let vision = null;
  let runtimeErrors = [];

  for (const source of CDN_SOURCES) {
    try {
      status(
        'Loading runtime from ' + source.name
      );

      vision =
        await FilesetResolver.forVisionTasks(
          source.wasm
        );

      break;

    } catch (error) {
      runtimeErrors.push(
        source.name + ': ' + errorMessage(error)
      );
      console.error(error);
    }
  }

  if (!vision) {
    throw new Error(
      'AI runtime failed: ' +
      runtimeErrors.join(' | ')
    );
  }

  if (!faceTask) {
    status('Loading face model...');

    faceTask =
      await FaceLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              `${MODEL_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
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

    handTask =
      await HandLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              `${MODEL_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: 'CPU'
          },
          runningMode: 'VIDEO',
          numHands: 2
        }
      );
  }

  status('AI models ready');
}

// ====================================
// CAMERA
// ====================================

async function startCamera() {
  if (running || starting) return;

  const currentSession = ++session;

  starting = true;
  updateButtons();

  try {
    if (!video || !canvas || !ctx) {
      throw new Error(
        'Missing video or canvas in index.html'
      );
    }

    if (!window.isSecureContext) {
      throw new Error(
        'Camera requires HTTPS'
      );
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        'Camera API unavailable in this browser'
      );
    }

    status('Requesting camera permission...');

    const newStream =
      await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: {
            ideal: front ? 'user' : 'environment'
          },
          width: { ideal: 640 },
          height: { ideal: 480 }
        }
      });

    if (currentSession !== session) {
      newStream.getTracks().forEach(
        track => track.stop()
      );
      return;
    }

    stream = newStream;

    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;

    await video.play();

    if (currentSession !== session) return;

    $('placeholder').hidden = true;

    running = true;
    starting = false;
    lastVideoTime = -1;

    updateButtons();

    status('Camera active — loading AI...');

    await initializeModels();

    if (
      !running ||
      currentSession !== session
    ) return;

    status('Analyzing locally');
    analyzeFrame(currentSession);

  } catch (error) {
    if (currentSession !== session) return;

    running = false;
    starting = false;

    cleanupCamera();
    updateButtons();
    showError(error);
  }
}

function cleanupCamera() {
  if (raf !== null) {
    cancelAnimationFrame(raf);
    raf = null;
  }

  if (stream) {
    stream.getTracks().forEach(
      track => track.stop()
    );
  }

  stream = null;

  if (video) {
    video.pause();
    video.srcObject = null;
  }

  if (ctx && canvas) {
    ctx.clearRect(
      0, 0,
      canvas.width,
      canvas.height
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
  session++;
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

// ====================================
// FACE GEOMETRY
// ====================================

function distance(a, b) {
  return Math.hypot(
    a.x - b.x,
    a.y - b.y
  );
}

function ratio(a, b, c, d) {
  return distance(a, b) /
    Math.max(
      0.00001,
      distance(c, d)
    );
}

function createFaceTemplate(l) {
  const pairs = [
    [33,263],
    [61,291],
    [1,152],
    [4,168],
    [133,362],
    [70,300],
    [234,454],
    [10,152],
    [13,14],
    [159,145],
    [386,374],
    [105,334],
    [58,288],
    [172,397]
  ];

  const scale =
    distance(l[33], l[263]) || 1;

  return pairs.map(([a,b]) =>
    Math.min(
      5,
      distance(l[a], l[b]) / scale
    )
  );
}

function compareTemplates(a, b) {
  if (
    !Array.isArray(a) ||
    !Array.isArray(b) ||
    !a.length ||
    a.length !== b.length
  ) return Infinity;

  return Math.sqrt(
    a.reduce(
      (sum, value, i) =>
        sum + (value - b[i]) ** 2,
      0
    ) / a.length
  );
}

function findClosestProfile(template) {
  if (
    !encryptionKey ||
    !profiles.length
  ) return null;

  let best = null;

  for (const p of profiles) {
    const score =
      compareTemplates(
        template,
        p.template
      );

    if (
      Number.isFinite(score) &&
      (!best || score < best.score)
    ) {
      best = {
        name: p.name,
        score
      };
    }
  }

  return best;
}

// ====================================
// HAND GESTURES
// ====================================

function detectGesture(l) {
  const wrist = l[0];

  const tips = [8,12,16,20];
  const joints = [6,10,14,18];

  const extended = tips.map(
    (tip, i) =>
      distance(l[tip], wrist) >
      distance(l[joints[i]], wrist) * 1.13
  );

  const count =
    extended.filter(Boolean).length;

  const pinch =
    distance(l[4], l[8]) <
    distance(l[5], l[17]) * 0.25;

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

// ====================================
// FACIAL MOVEMENT
// ====================================

function analyzeFace(l, blendshapes) {
  const width =
    distance(l[33], l[263]) || 1;

  const offset =
    (
      l[1].x -
      (l[33].x + l[263].x) / 2
    ) / width;

  const head =
    offset > 0.10
      ? 'Right'
      : offset < -0.10
        ? 'Left'
        : 'Center';

  const leftEye = ratio(
    l[159], l[145],
    l[33], l[133]
  );

  const rightEye = ratio(
    l[386], l[374],
    l[362], l[263]
  );

  const mouth = ratio(
    l[13], l[14],
    l[61], l[291]
  );

  const scores = Object.fromEntries(
    (blendshapes || []).map(
      item => [
        item.categoryName,
        item.score
      ]
    )
  );

  const blink =
    (
      (scores.eyeBlinkLeft ?? 0) > 0.55 &&
      (scores.eyeBlinkRight ?? 0) > 0.55
    ) ||
    (
      leftEye < 0.12 &&
      rightEye < 0.12
    );

  const mouthOpen = mouth > 0.10;

  const centerY =
    (l[1].y + l[152].y) / 2;

  let motion = '';
  const now = performance.now();

  if (
    previousHead &&
    now - previousHead.time < 1000
  ) {
    const change =
      centerY - previousHead.y;

    if (Math.abs(change) > 0.025) {
      motion =
        change > 0
          ? ' · Moving down'
          : ' · Moving up';
    }
  }

  previousHead = {
    y: centerY,
    time: now
  };

  return (
    `Head: ${head} · ` +
    `Eyes: ${blink ? 'Blink/closed' : 'Open'} · ` +
    `Mouth: ${mouthOpen ? 'Open' : 'Closed'}` +
    motion
  );
}

// ====================================
// DRAWING
// ====================================

function drawLandmarks(points, color) {
  if (!ctx) return;

  ctx.fillStyle = color;

  for (const p of points) {
    const x =
      (front ? 1 - p.x : p.x) *
      canvas.width;

    const y = p.y * canvas.height;

    ctx.beginPath();
    ctx.arc(x, y, 1.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ====================================
// REAL-TIME ANALYSIS
// ====================================

function analyzeFrame(currentSession) {
  if (
    !running ||
    currentSession !== session
  ) return;

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
        faceTask.detectForVideo(
          video,
          now
        );

      const handResults =
        handTask.detectForVideo(
          video,
          now
        );

      ctx.clearRect(
        0, 0,
        canvas.width,
        canvas.height
      );

      const faces =
        faceResults.faceLandmarks || [];

      const hands =
        handResults.landmarks || [];

      lastFaceCount = faces.length;

      lastFace =
        faces.length === 1
          ? createFaceTemplate(faces[0])
          : null;

      faces.forEach(
        face => drawLandmarks(
          face,
          '#5df4bd'
        )
      );

      hands.forEach(
        hand => drawLandmarks(
          hand,
          '#ffc970'
        )
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
            ? ' · ' +
              hands.map(
                detectGesture
              ).join(' / ')
            : ''
        )
      );

      if (faces.length === 1) {
        const blend =
          faceResults.faceBlendshapes?.[0]
            ?.categories;

        setText(
          'motionInfo',
          analyzeFace(
            faces[0],
            blend
          )
        );

        const match =
          findClosestProfile(lastFace);

        setText(
          'matchInfo',
          match
            ? `Closest saved geometry: ${match.name} ` +
              `(distance ${match.score.toFixed(3)}; ` +
              'identity unverified)'
            : encryptionKey
              ? 'No saved profiles'
              : 'Face library locked'
        );

      } else {
        setText(
          'motionInfo',
          'Movement: —'
        );

        setText(
          'matchInfo',
          faces.length > 1
            ? 'Matching paused: multiple faces'
            : 'Saved face: —'
        );

        previousHead = null;
      }

      updateButtons();
    }

  } catch (error) {
    const message = errorMessage(error);

    if (message !== lastError) {
      lastError = message;
      console.error(error);

      status(
        'Analysis error: ' + message
      );
    }
  }

  raf = requestAnimationFrame(
    () => analyzeFrame(currentSession)
  );
}

// ====================================
// INDEXEDDB
// ====================================

function openDatabase() {
  return new Promise(
    (resolve, reject) => {
      const request = indexedDB.open(
        'vision-ai-private-v1',
        1
      );

      request.onupgradeneeded = () => {
        const database = request.result;

        if (
          !database.objectStoreNames
            .contains('data')
        ) {
          database.createObjectStore(
            'data'
          );
        }
      };

      request.onsuccess = () =>
        resolve(request.result);

      request.onerror = () =>
        reject(request.error);
    }
  );
}

function dbGet(name) {
  return new Promise(
    (resolve, reject) => {
      const request = db
        .transaction('data')
        .objectStore('data')
        .get(name);

      request.onsuccess = () =>
        resolve(request.result);

      request.onerror = () =>
        reject(request.error);
    }
  );
}

function dbPut(name, value) {
  return new Promise(
    (resolve, reject) => {
      const request = db
        .transaction(
          'data',
          'readwrite'
        )
        .objectStore('data')
        .put(value, name);

      request.onsuccess = () =>
        resolve();

      request.onerror = () =>
        reject(request.error);
    }
  );
}

// ====================================
// ENCRYPTION
// ====================================

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
    c => c.charCodeAt(0)
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
      {
        name: 'AES-GCM',
        iv
      },
      encryptionKey,
      encoder.encode(
        JSON.stringify(data)
      )
    );

  return {
    iv: toBase64(iv),
    cipher: toBase64(encrypted)
  };
}

async function saveLibrary() {
  if (!db || !encryptionKey) {
    throw new Error(
      'Face library is locked'
    );
  }

  await dbPut(
    'vault',
    await encryptProfiles(profiles)
  );

  renderProfiles();
}

// ====================================
// SAVED PROFILES
// ====================================

function renderProfiles() {
  const list = $('people');
  if (!list) return;

  list.replaceChildren();

  for (const profile of profiles) {
    const item =
      document.createElement('li');

    const name =
      document.createElement('span');

    const button =
      document.createElement('button');

    name.textContent = profile.name;
    button.textContent = 'Delete';

    button.onclick = async () => {
      if (
        !confirm(
          `Delete ${profile.name}?`
        )
      ) return;

      const previous = profiles;

      try {
        profiles = profiles.filter(
          p => p.id !== profile.id
        );

        await saveLibrary();

        libraryStatus(
          'Profile deleted'
        );

      } catch (error) {
        profiles = previous;
        renderProfiles();

        libraryStatus(
          'Delete failed: ' +
          errorMessage(error)
        );
      }
    };

    item.append(name, button);
    list.append(item);
  }
}

// ====================================
// UNLOCK LIBRARY
// ====================================

async function unlockLibrary() {
  try {
    const passphrase =
      $('pass')?.value || '';

    if (passphrase.length < 10) {
      throw new Error(
        'Passphrase must have at least 10 characters'
      );
    }

    db ??= await openDatabase();

    const storedSalt =
      await dbGet('salt');

    const existing = !!storedSalt;

    const salt = existing
      ? fromBase64(storedSalt)
      : crypto.getRandomValues(
          new Uint8Array(16)
        );

    const candidateKey =
      await deriveKey(
        passphrase,
        salt
      );

    const vault =
      await dbGet('vault');

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

      loadedProfiles =
        JSON.parse(
          decoder.decode(decrypted)
        );

      if (
        !Array.isArray(loadedProfiles)
      ) {
        throw new Error(
          'Invalid library data'
        );
      }

    } else if (existing) {
      throw new Error(
        'Encrypted library data missing'
      );
    }

    encryptionKey = candidateKey;
    profiles = loadedProfiles;

    if (!existing) {
      await saveLibrary();

      await dbPut(
        'salt',
        toBase64(salt)
      );
    }

    $('pass').value = '';

    renderProfiles();
    updateButtons();

    libraryStatus(
      `Unlocked · ${profiles.length} profiles`
    );

  } catch (error) {
    encryptionKey = null;
    profiles = [];

    renderProfiles();
    updateButtons();

    libraryStatus(
      'Unlock failed: ' +
      errorMessage(error)
    );
  }
}

// ====================================
// LOCK LIBRARY
// ====================================

function lockLibrary() {
  encryptionKey = null;
  profiles = [];

  renderProfiles();
  updateButtons();

  libraryStatus('Library locked');
}

// ====================================
// SAVE FACE
// ====================================

async function saveFace() {
  try {
    if (
      !encryptionKey ||
      !lastFace ||
      lastFaceCount !== 1
    ) {
      throw new Error(
        'Unlock library and show exactly one face'
      );
    }

    if (!$('consent')?.checked) {
      throw new Error(
        'Consent confirmation required'
      );
    }

    const name =
      $('person')?.value.trim() || '';

    if (
      !name ||
      name.length > 80
    ) {
      throw new Error(
        'Enter a name of 1–80 characters'
      );
    }

    const profile = {
      id: crypto.randomUUID(),
      name,
      template: [...lastFace],
      createdAt:
        new Date().toISOString()
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
      `${name} saved locally (experimental geometry)`
    );

  } catch (error) {
    libraryStatus(
      errorMessage(error)
    );
  }
}

// ====================================
// INITIALIZE BUTTONS
// ====================================

function initializeApp() {
  const start = $('start');

  if (!start) {
    status(
      'Error: Start button missing'
    );
    return;
  }

  start.onclick = startCamera;

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

  status(
    'Vision AI ready — tap Start camera'
  );

  libraryStatus(
    'Library locked'
  );
}

// ====================================
// GLOBAL ERROR HANDLING
// ====================================

window.addEventListener(
  'error',
  event => {
    console.error(event.error);

    status(
      'JavaScript error: ' +
      event.message
    );
  }
);

window.addEventListener(
  'unhandledrejection',
  event => {
    const message =
      errorMessage(event.reason);

    console.error(event.reason);

    status(
      'Error: ' + message
    );
  }
);

window.addEventListener(
  'pagehide',
  () => {
    session++;
    running = false;
    starting = false;

    cleanupCamera();

    encryptionKey = null;
    profiles = [];
  }
);

// ====================================
// START APPLICATION
// ====================================

if (document.readyState === 'loading') {
  document.addEventListener(
    'DOMContentLoaded',
    initializeApp,
    { once: true }
  );
} else {
  initializeApp();
}
