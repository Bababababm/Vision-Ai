
/*
 Vision AI — Mobile Web App
 Replace the entire app.js with this file.

 Features:
 - Camera startup with visible errors
 - Face landmarks
 - Hand gesture estimates
 - Head, eye and mouth movement estimates
 - Experimental saved-face geometry comparison
 - AES-GCM encrypted IndexedDB profiles
*/

const $ = id => document.getElementById(id);

const video = $('video');
const canvas = $('overlay');
const ctx = canvas?.getContext('2d');

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
let lastVideoTime = -1;
let lastError = '';

let lastFace = null;
let lastFaceCount = 0;
let previousHead = null;

let db = null;
let key = null;
let profiles = [];

const MP_VERSION = '0.10.22';

const MP_CDN =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;

const MODEL_BASE =
  'https://storage.googleapis.com/mediapipe-models';

const enc = new TextEncoder();
const dec = new TextDecoder();

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function status(message) {
  setText('status', message);
}

function libraryStatus(message) {
  setText('libraryStatus', message);
}

function showError(error) {
  const message = error?.message || String(error);

  console.error('Vision AI:', error);
  status('Error: ' + message);

  alert('Vision AI error:\n' + message);
}

function updateButtons() {
  if ($('start')) {
    $('start').disabled = running || starting;
  }

  if ($('stop')) {
    $('stop').disabled = !running && !starting;
  }

  if ($('switch')) {
    $('switch').disabled = starting;
  }

  if ($('unlock')) {
    $('unlock').disabled = !!key;
  }

  if ($('lock')) {
    $('lock').disabled = !key;
  }

  if ($('save')) {
    $('save').disabled =
      !running ||
      !key ||
      !lastFace ||
      lastFaceCount !== 1;
  }
}

// -----------------------------
// AI model initialization
// -----------------------------

async function initModels() {
  if (faceTask && handTask) return;

  status('Downloading AI library...');

  if (!FilesetResolver) {
    const mp = await import(`${MP_CDN}/+esm`);

    FaceLandmarker = mp.FaceLandmarker;
    HandLandmarker = mp.HandLandmarker;
    FilesetResolver = mp.FilesetResolver;

    if (
      !FaceLandmarker ||
      !HandLandmarker ||
      !FilesetResolver
    ) {
      throw new Error(
        'MediaPipe library did not load correctly.'
      );
    }
  }

  status('Loading AI runtime...');

  const files =
    await FilesetResolver.forVisionTasks(
      `${MP_CDN}/wasm`
    );

  status('Loading face model...');

  if (!faceTask) {
    faceTask =
      await FaceLandmarker.createFromOptions(
        files,
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

  status('Loading hand model...');

  if (!handTask) {
    handTask =
      await HandLandmarker.createFromOptions(
        files,
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

// -----------------------------
// Camera
// -----------------------------

async function start() {
  if (running || starting) return;

  starting = true;
  updateButtons();

  try {
    if (!video || !canvas || !ctx) {
      throw new Error(
        'Missing video or overlay element in index.html.'
      );
    }

    if (!window.isSecureContext) {
      throw new Error(
        'Camera requires HTTPS. Open the GitHub Pages website.'
      );
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        'Camera API unavailable. Try opening this site in Chrome.'
      );
    }

    // Request camera before downloading models.
    // This makes permission failures visible immediately.
    status('Requesting camera permission...');

    stream =
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

    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;

    await video.play();

    if ($('placeholder')) {
      $('placeholder').hidden = true;
    }

    running = true;
    starting = false;
    lastVideoTime = -1;

    status('Camera active. Loading AI...');
    updateButtons();

    // Start displaying the camera before AI loads.
    await initModels();

    if (!running) return;

    status('Analyzing locally');
    loop();

  } catch (error) {
    cleanupCamera();
    starting = false;
    running = false;
    updateButtons();
    showError(error);
  }
}

function cleanupCamera() {
  cancelAnimationFrame(raf);
  raf = null;

  if (stream) {
    stream.getTracks().forEach(track => {
      track.stop();
    });
  }

  stream = null;

  if (video) {
    video.pause();
    video.srcObject = null;
  }

  if (ctx && canvas) {
    ctx.clearRect(
      0,
      0,
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

function stop() {
  running = false;
  starting = false;

  cleanupCamera();
  updateButtons();

  status('Stopped');
}

async function switchCamera() {
  if (starting) return;

  const wasRunning = running;

  stop();
  front = !front;

  if (wasRunning) {
    await start();
  } else {
    status(
      front
        ? 'Front camera selected'
        : 'Rear camera selected'
    );
  }
}

// -----------------------------
// Geometry utilities
// -----------------------------

function dist(a, b) {
  return Math.hypot(
    a.x - b.x,
    a.y - b.y
  );
}

function ratio(a, b, c, d) {
  return dist(a, b) /
    Math.max(0.00001, dist(c, d));
}

// Experimental landmark geometry.
// This is NOT a reliable face identity model.
function template(l) {
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
    dist(l[33], l[263]) || 1;

  return pairs.map(([a, b]) =>
    Math.min(
      5,
      dist(l[a], l[b]) / scale
    )
  );
}

function profileDistance(a, b) {
  if (
    !Array.isArray(a) ||
    !Array.isArray(b) ||
    a.length !== b.length ||
    !a.length
  ) {
    return Infinity;
  }

  return Math.sqrt(
    a.reduce(
      (sum, value, i) =>
        sum + (value - b[i]) ** 2,
      0
    ) / a.length
  );
}

function closestProfile(t) {
  if (!key || !profiles.length) {
    return null;
  }

  let best = null;

  for (const p of profiles) {
    const distance =
      profileDistance(t, p.template);

    if (
      Number.isFinite(distance) &&
      (!best || distance < best.distance)
    ) {
      best = {
        name: p.name,
        distance
      };
    }
  }

  return best;
}

// -----------------------------
// Hand gesture estimates
// -----------------------------

function gesture(l) {
  const wrist = l[0];

  const tips = [8, 12, 16, 20];
  const pips = [6, 10, 14, 18];

  const extended = tips.map(
    (id, i) =>
      dist(l[id], wrist) >
      dist(l[pips[i]], wrist) * 1.13
  );

  const count =
    extended.filter(Boolean).length;

  const pinch =
    dist(l[4], l[8]) <
    dist(l[5], l[17]) * 0.25;

  if (pinch) return 'Pinch';

  if (count === 4) {
    return 'Open palm';
  }

  if (count === 0) {
    return 'Fist';
  }

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

// -----------------------------
// Face movement analysis
// -----------------------------

function faceAnalysis(l, blend) {
  const width =
    dist(l[33], l[263]) || 1;

  const offset =
    (
      l[1].x -
      (l[33].x + l[263].x) / 2
    ) / width;

  const yaw =
    offset > 0.10
      ? 'turned right'
      : offset < -0.10
        ? 'turned left'
        : 'centered';

  const eyeL = ratio(
    l[159],
    l[145],
    l[33],
    l[133]
  );

  const eyeR = ratio(
    l[386],
    l[374],
    l[362],
    l[263]
  );

  const mouth = ratio(
    l[13],
    l[14],
    l[61],
    l[291]
  );

  const scores = Object.fromEntries(
    (blend || []).map(item => [
      item.categoryName,
      item.score
    ])
  );

  const blink =
    (
      (scores.eyeBlinkLeft ?? 0) > 0.55 &&
      (scores.eyeBlinkRight ?? 0) > 0.55
    ) ||
    (
      eyeL < 0.12 &&
      eyeR < 0.12
    );

  const mouthOpen = mouth > 0.10;

  const centerY =
    (l[1].y + l[152].y) / 2;

  let motion = '—';

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
          ? 'Head moving down'
          : 'Head moving up';
    }
  }

  previousHead = {
    y: centerY,
    time: now
  };

  return {
    text:
      `Head: ${yaw} · ` +
      `Eyes: ${blink ? 'blink/closed' : 'open'} · ` +
      `Mouth: ${mouthOpen ? 'open' : 'closed'}`,
    motion
  };
}

// -----------------------------
// Drawing
// -----------------------------

function drawLandmarks(landmarks, color) {
  if (!ctx || !canvas) return;

  ctx.fillStyle = color;

  for (const p of landmarks) {
    const x =
      (front ? 1 - p.x : p.x) *
      canvas.width;

    const y =
      p.y * canvas.height;

    ctx.beginPath();
    ctx.arc(x, y, 1.4, 0, Math.PI * 2);
    ctx.fill();
  }
}

// -----------------------------
// Real-time processing
// -----------------------------

function loop() {
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

      const faces =
        faceTask.detectForVideo(video, now);

      const hands =
        handTask.detectForVideo(video, now);

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );

      const fl =
        faces.faceLandmarks || [];

      const hl =
        hands.landmarks || [];

      lastFaceCount = fl.length;

      lastFace =
        fl.length === 1
          ? template(fl[0])
          : null;

      fl.forEach(l =>
        drawLandmarks(l, '#5df4bd')
      );

      hl.forEach(l =>
        drawLandmarks(l, '#ffc970')
      );

      setText(
        'faceInfo',
        `Faces: ${fl.length}`
      );

      setText(
        'handInfo',
        `Hands: ${hl.length}` +
        (
          hl.length
            ? ' · ' +
              hl.map(gesture).join(' / ')
            : ''
        )
      );

      if (fl.length === 1) {
        const info = faceAnalysis(
          fl[0],
          faces.faceBlendshapes?.[0]
            ?.categories
        );

        setText(
          'motionInfo',
          info.text +
          (
            info.motion !== '—'
              ? ' · ' + info.motion
              : ''
          )
        );

        const m =
          closestProfile(lastFace);

        if (m) {
          setText(
            'matchInfo',
            `Closest saved geometry: ${m.name} ` +
            `(distance ${m.distance.toFixed(3)}; ` +
            'identity not verified)'
          );
        } else {
          setText(
            'matchInfo',
            key
              ? 'No saved profiles'
              : 'Face library locked'
          );
        }

      } else {
        setText(
          'motionInfo',
          'Movement: —'
        );

        setText(
          'matchInfo',
          fl.length > 1
            ? 'Matching paused: multiple faces'
            : 'Saved face: —'
        );

        previousHead = null;
      }

      updateButtons();
    }

    if (lastError) {
      lastError = '';
    }

  } catch (error) {
    const message =
      error?.message || String(error);

    if (message !== lastError) {
      lastError = message;
      console.error(error);

      status(
        'Analysis error: ' + message
      );
    }
  }

  raf = requestAnimationFrame(loop);
}

// -----------------------------
// IndexedDB storage
// -----------------------------

function openDb() {
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
          database.createObjectStore('data');
        }
      };

      request.onsuccess = () =>
        resolve(request.result);

      request.onerror = () =>
        reject(request.error);
    }
  );
}

function get(name) {
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

function put(name, value) {
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

// -----------------------------
// Encryption
// -----------------------------

function b64(value) {
  return btoa(
    String.fromCharCode(
      ...new Uint8Array(value)
    )
  );
}

function bytes(value) {
  return Uint8Array.from(
    atob(value),
    c => c.charCodeAt(0)
  );
}

async function derive(pass, salt) {
  const material =
    await crypto.subtle.importKey(
      'raw',
      enc.encode(pass),
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

async function encryptData(data) {
  const iv =
    crypto.getRandomValues(
      new Uint8Array(12)
    );

  const cipher =
    await crypto.subtle.encrypt(
      {
        name: 'AES-GCM',
        iv
      },
      key,
      enc.encode(JSON.stringify(data))
    );

  return {
    iv: b64(iv),
    cipher: b64(cipher)
  };
}

async function persist() {
  if (!key || !db) {
    throw new Error(
      'Face library is locked.'
    );
  }

  await put(
    'vault',
    await encryptData(profiles)
  );

  renderPeople();
}

// -----------------------------
// Saved profile display
// -----------------------------

function renderPeople() {
  const ul = $('people');
  if (!ul) return;

  ul.replaceChildren();

  for (const p of profiles) {
    const li =
      document.createElement('li');

    const name =
      document.createElement('span');

    const del =
      document.createElement('button');

    name.textContent = p.name;
    del.textContent = 'Delete';

    del.onclick = async () => {
      if (
        !confirm(
          `Delete ${p.name}'s saved profile?`
        )
      ) {
        return;
      }

      const previous = profiles;

      try {
        profiles = profiles.filter(
          x => x.id !== p.id
        );

        await persist();

        libraryStatus(
          'Profile deleted'
        );

      } catch (error) {
        profiles = previous;
        renderPeople();

        libraryStatus(
          'Delete failed: ' + error.message
        );
      }
    };

    li.append(name, del);
    ul.append(li);
  }
}

// -----------------------------
// Unlock encrypted library
// -----------------------------

async function unlock() {
  try {
    const pass = $('pass')?.value || '';

    if (pass.length < 10) {
      throw new Error(
        'Use a passphrase of at least 10 characters.'
      );
    }

    db ??= await openDb();

    const storedSalt =
      await get('salt');

    const existing = !!storedSalt;

    const proposedSalt = existing
      ? bytes(storedSalt)
      : crypto.getRandomValues(
          new Uint8Array(16)
        );

    const proposedKey =
      await derive(
        pass,
        proposedSalt
      );

    const vault =
      await get('vault');

    let data = [];

    if (vault) {
      const plaintext =
        await crypto.subtle.decrypt(
          {
            name: 'AES-GCM',
            iv: bytes(vault.iv)
          },
          proposedKey,
          bytes(vault.cipher)
        );

      data = JSON.parse(
        dec.decode(plaintext)
      );

      if (!Array.isArray(data)) {
        throw new Error(
          'Invalid library format.'
        );
      }

    } else if (existing) {
      throw new Error(
        'Library data is missing.'
      );
    }

    key = proposedKey;
    profiles = data;

    if (!existing) {
      // Save encrypted empty vault first,
      // then store its salt.
      await persist();
      await put(
        'salt',
        b64(proposedSalt)
      );
    }

    $('pass').value = '';

    renderPeople();
    updateButtons();

    libraryStatus(
      `Library unlocked · ` +
      `${profiles.length} saved profiles`
    );

  } catch (error) {
    key = null;
    profiles = [];

    renderPeople();
    updateButtons();

    libraryStatus(
      'Could not unlock: ' +
      error.message
    );
  }
}

// -----------------------------
// Lock library
// -----------------------------

function lock() {
  key = null;
  profiles = [];

  renderPeople();
  updateButtons();

  libraryStatus('Library locked');
}

// -----------------------------
// Save a consenting person's
// experimental geometry profile
// -----------------------------

async function saveFace() {
  try {
    if (
      !key ||
      !lastFace ||
      lastFaceCount !== 1
    ) {
      throw new Error(
        'Unlock the library and show exactly one face.'
      );
    }

    if (!$('consent')?.checked) {
      throw new Error(
        'Consent confirmation required.'
      );
    }

    const name =
      $('person')?.value.trim() || '';

    if (
      !name ||
      name.length > 80
    ) {
      throw new Error(
        'Enter a name of 1–80 characters.'
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
      await persist();
    } catch (error) {
      profiles = profiles.filter(
        p => p.id !== profile.id
      );

      throw error;
    }

    $('person').value = '';
    $('consent').checked = false;

    libraryStatus(
      `${name} saved locally ` +
      '(experimental face geometry)'
    );

  } catch (error) {
    libraryStatus(error.message);
  }
}

// -----------------------------
// Connect buttons immediately
// -----------------------------

function initializeApp() {
  const required = [
    'video',
    'overlay',
    'start',
    'stop',
    'switch',
    'unlock',
    'lock',
    'save',
    'pass',
    'person',
    'consent',
    'people'
  ];

  const missing =
    required.filter(id => !$(id));

  if (missing.length) {
    const message =
      'Missing HTML elements: ' +
      missing.join(', ');

    console.error(message);
    status(message);
    return;
  }

  $('start').addEventListener(
    'click',
    start
  );

  $('stop').addEventListener(
    'click',
    stop
  );

  $('switch').addEventListener(
    'click',
    switchCamera
  );

  $('unlock').addEventListener(
    'click',
    unlock
  );

  $('lock').addEventListener(
    'click',
    lock
  );

  $('save').addEventListener(
    'click',
    saveFace
  );

  updateButtons();

  status('Idle — ready to start');
  libraryStatus('Library locked');
}

window.addEventListener(
  'pagehide',
  () => {
    running = false;
    cleanupCamera();

    key = null;
    profiles = [];
  }
);

if (document.readyState === 'loading') {
  document.addEventListener(
    'DOMContentLoaded',
    initializeApp
  );
} else {
  initializeApp();
}
/*
 Vision AI — Mobile Web App
 Replace the entire app.js with this file.

 Features:
 - Camera startup with visible errors
 - Face landmarks
 - Hand gesture estimates
 - Head, eye and mouth movement estimates
 - Experimental saved-face geometry comparison
 - AES-GCM encrypted IndexedDB profiles
*/

const $ = id => document.getElementById(id);

const video = $('video');
const canvas = $('overlay');
const ctx = canvas?.getContext('2d');

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
let lastVideoTime = -1;
let lastError = '';

let lastFace = null;
let lastFaceCount = 0;
let previousHead = null;

let db = null;
let key = null;
let profiles = [];

const MP_VERSION = '0.10.22';

const MP_CDN =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;

const MODEL_BASE =
  'https://storage.googleapis.com/mediapipe-models';

const enc = new TextEncoder();
const dec = new TextDecoder();

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function status(message) {
  setText('status', message);
}

function libraryStatus(message) {
  setText('libraryStatus', message);
}

function showError(error) {
  const message = error?.message || String(error);

  console.error('Vision AI:', error);
  status('Error: ' + message);

  alert('Vision AI error:\n' + message);
}

function updateButtons() {
  if ($('start')) {
    $('start').disabled = running || starting;
  }

  if ($('stop')) {
    $('stop').disabled = !running && !starting;
  }

  if ($('switch')) {
    $('switch').disabled = starting;
  }

  if ($('unlock')) {
    $('unlock').disabled = !!key;
  }

  if ($('lock')) {
    $('lock').disabled = !key;
  }

  if ($('save')) {
    $('save').disabled =
      !running ||
      !key ||
      !lastFace ||
      lastFaceCount !== 1;
  }
}

// -----------------------------
// AI model initialization
// -----------------------------

async function initModels() {
  if (faceTask && handTask) return;

  status('Downloading AI library...');

  if (!FilesetResolver) {
    const mp = await import(`${MP_CDN}/+esm`);

    FaceLandmarker = mp.FaceLandmarker;
    HandLandmarker = mp.HandLandmarker;
    FilesetResolver = mp.FilesetResolver;

    if (
      !FaceLandmarker ||
      !HandLandmarker ||
      !FilesetResolver
    ) {
      throw new Error(
        'MediaPipe library did not load correctly.'
      );
    }
  }

  status('Loading AI runtime...');

  const files =
    await FilesetResolver.forVisionTasks(
      `${MP_CDN}/wasm`
    );

  status('Loading face model...');

  if (!faceTask) {
    faceTask =
      await FaceLandmarker.createFromOptions(
        files,
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

  status('Loading hand model...');

  if (!handTask) {
    handTask =
      await HandLandmarker.createFromOptions(
        files,
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

// -----------------------------
// Camera
// -----------------------------

async function start() {
  if (running || starting) return;

  starting = true;
  updateButtons();

  try {
    if (!video || !canvas || !ctx) {
      throw new Error(
        'Missing video or overlay element in index.html.'
      );
    }

    if (!window.isSecureContext) {
      throw new Error(
        'Camera requires HTTPS. Open the GitHub Pages website.'
      );
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        'Camera API unavailable. Try opening this site in Chrome.'
      );
    }

    // Request camera before downloading models.
    // This makes permission failures visible immediately.
    status('Requesting camera permission...');

    stream =
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

    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;

    await video.play();

    if ($('placeholder')) {
      $('placeholder').hidden = true;
    }

    running = true;
    starting = false;
    lastVideoTime = -1;

    status('Camera active. Loading AI...');
    updateButtons();

    // Start displaying the camera before AI loads.
    await initModels();

    if (!running) return;

    status('Analyzing locally');
    loop();

  } catch (error) {
    cleanupCamera();
    starting = false;
    running = false;
    updateButtons();
    showError(error);
  }
}

function cleanupCamera() {
  cancelAnimationFrame(raf);
  raf = null;

  if (stream) {
    stream.getTracks().forEach(track => {
      track.stop();
    });
  }

  stream = null;

  if (video) {
    video.pause();
    video.srcObject = null;
  }

  if (ctx && canvas) {
    ctx.clearRect(
      0,
      0,
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

function stop() {
  running = false;
  starting = false;

  cleanupCamera();
  updateButtons();

  status('Stopped');
}

async function switchCamera() {
  if (starting) return;

  const wasRunning = running;

  stop();
  front = !front;

  if (wasRunning) {
    await start();
  } else {
    status(
      front
        ? 'Front camera selected'
        : 'Rear camera selected'
    );
  }
}

// -----------------------------
// Geometry utilities
// -----------------------------

function dist(a, b) {
  return Math.hypot(
    a.x - b.x,
    a.y - b.y
  );
}

function ratio(a, b, c, d) {
  return dist(a, b) /
    Math.max(0.00001, dist(c, d));
}

// Experimental landmark geometry.
// This is NOT a reliable face identity model.
function template(l) {
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
    dist(l[33], l[263]) || 1;

  return pairs.map(([a, b]) =>
    Math.min(
      5,
      dist(l[a], l[b]) / scale
    )
  );
}

function profileDistance(a, b) {
  if (
    !Array.isArray(a) ||
    !Array.isArray(b) ||
    a.length !== b.length ||
    !a.length
  ) {
    return Infinity;
  }

  return Math.sqrt(
    a.reduce(
      (sum, value, i) =>
        sum + (value - b[i]) ** 2,
      0
    ) / a.length
  );
}

function closestProfile(t) {
  if (!key || !profiles.length) {
    return null;
  }

  let best = null;

  for (const p of profiles) {
    const distance =
      profileDistance(t, p.template);

    if (
      Number.isFinite(distance) &&
      (!best || distance < best.distance)
    ) {
      best = {
        name: p.name,
        distance
      };
    }
  }

  return best;
}

// -----------------------------
// Hand gesture estimates
// -----------------------------

function gesture(l) {
  const wrist = l[0];

  const tips = [8, 12, 16, 20];
  const pips = [6, 10, 14, 18];

  const extended = tips.map(
    (id, i) =>
      dist(l[id], wrist) >
      dist(l[pips[i]], wrist) * 1.13
  );

  const count =
    extended.filter(Boolean).length;

  const pinch =
    dist(l[4], l[8]) <
    dist(l[5], l[17]) * 0.25;

  if (pinch) return 'Pinch';

  if (count === 4) {
    return 'Open palm';
  }

  if (count === 0) {
    return 'Fist';
  }

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

// -----------------------------
// Face movement analysis
// -----------------------------

function faceAnalysis(l, blend) {
  const width =
    dist(l[33], l[263]) || 1;

  const offset =
    (
      l[1].x -
      (l[33].x + l[263].x) / 2
    ) / width;

  const yaw =
    offset > 0.10
      ? 'turned right'
      : offset < -0.10
        ? 'turned left'
        : 'centered';

  const eyeL = ratio(
    l[159],
    l[145],
    l[33],
    l[133]
  );

  const eyeR = ratio(
    l[386],
    l[374],
    l[362],
    l[263]
  );

  const mouth = ratio(
    l[13],
    l[14],
    l[61],
    l[291]
  );

  const scores = Object.fromEntries(
    (blend || []).map(item => [
      item.categoryName,
      item.score
    ])
  );

  const blink =
    (
      (scores.eyeBlinkLeft ?? 0) > 0.55 &&
      (scores.eyeBlinkRight ?? 0) > 0.55
    ) ||
    (
      eyeL < 0.12 &&
      eyeR < 0.12
    );

  const mouthOpen = mouth > 0.10;

  const centerY =
    (l[1].y + l[152].y) / 2;

  let motion = '—';

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
          ? 'Head moving down'
          : 'Head moving up';
    }
  }

  previousHead = {
    y: centerY,
    time: now
  };

  return {
    text:
      `Head: ${yaw} · ` +
      `Eyes: ${blink ? 'blink/closed' : 'open'} · ` +
      `Mouth: ${mouthOpen ? 'open' : 'closed'}`,
    motion
  };
}

// -----------------------------
// Drawing
// -----------------------------

function drawLandmarks(landmarks, color) {
  if (!ctx || !canvas) return;

  ctx.fillStyle = color;

  for (const p of landmarks) {
    const x =
      (front ? 1 - p.x : p.x) *
      canvas.width;

    const y =
      p.y * canvas.height;

    ctx.beginPath();
    ctx.arc(x, y, 1.4, 0, Math.PI * 2);
    ctx.fill();
  }
}

// -----------------------------
// Real-time processing
// -----------------------------

function loop() {
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

      const faces =
        faceTask.detectForVideo(video, now);

      const hands =
        handTask.detectForVideo(video, now);

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );

      const fl =
        faces.faceLandmarks || [];

      const hl =
        hands.landmarks || [];

      lastFaceCount = fl.length;

      lastFace =
        fl.length === 1
          ? template(fl[0])
          : null;

      fl.forEach(l =>
        drawLandmarks(l, '#5df4bd')
      );

      hl.forEach(l =>
        drawLandmarks(l, '#ffc970')
      );

      setText(
        'faceInfo',
        `Faces: ${fl.length}`
      );

      setText(
        'handInfo',
        `Hands: ${hl.length}` +
        (
          hl.length
            ? ' · ' +
              hl.map(gesture).join(' / ')
            : ''
        )
      );

      if (fl.length === 1) {
        const info = faceAnalysis(
          fl[0],
          faces.faceBlendshapes?.[0]
            ?.categories
        );

        setText(
          'motionInfo',
          info.text +
          (
            info.motion !== '—'
              ? ' · ' + info.motion
              : ''
          )
        );

        const m =
          closestProfile(lastFace);

        if (m) {
          setText(
            'matchInfo',
            `Closest saved geometry: ${m.name} ` +
            `(distance ${m.distance.toFixed(3)}; ` +
            'identity not verified)'
          );
        } else {
          setText(
            'matchInfo',
            key
              ? 'No saved profiles'
              : 'Face library locked'
          );
        }

      } else {
        setText(
          'motionInfo',
          'Movement: —'
        );

        setText(
          'matchInfo',
          fl.length > 1
            ? 'Matching paused: multiple faces'
            : 'Saved face: —'
        );

        previousHead = null;
      }

      updateButtons();
    }

    if (lastError) {
      lastError = '';
    }

  } catch (error) {
    const message =
      error?.message || String(error);

    if (message !== lastError) {
      lastError = message;
      console.error(error);

      status(
        'Analysis error: ' + message
      );
    }
  }

  raf = requestAnimationFrame(loop);
}

// -----------------------------
// IndexedDB storage
// -----------------------------

function openDb() {
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
          database.createObjectStore('data');
        }
      };

      request.onsuccess = () =>
        resolve(request.result);

      request.onerror = () =>
        reject(request.error);
    }
  );
}

function get(name) {
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

function put(name, value) {
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

// -----------------------------
// Encryption
// -----------------------------

function b64(value) {
  return btoa(
    String.fromCharCode(
      ...new Uint8Array(value)
    )
  );
}

function bytes(value) {
  return Uint8Array.from(
    atob(value),
    c => c.charCodeAt(0)
  );
}

async function derive(pass, salt) {
  const material =
    await crypto.subtle.importKey(
      'raw',
      enc.encode(pass),
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

async function encryptData(data) {
  const iv =
    crypto.getRandomValues(
      new Uint8Array(12)
    );

  const cipher =
    await crypto.subtle.encrypt(
      {
        name: 'AES-GCM',
        iv
      },
      key,
      enc.encode(JSON.stringify(data))
    );

  return {
    iv: b64(iv),
    cipher: b64(cipher)
  };
}

async function persist() {
  if (!key || !db) {
    throw new Error(
      'Face library is locked.'
    );
  }

  await put(
    'vault',
    await encryptData(profiles)
  );

  renderPeople();
}

// -----------------------------
// Saved profile display
// -----------------------------

function renderPeople() {
  const ul = $('people');
  if (!ul) return;

  ul.replaceChildren();

  for (const p of profiles) {
    const li =
      document.createElement('li');

    const name =
      document.createElement('span');

    const del =
      document.createElement('button');

    name.textContent = p.name;
    del.textContent = 'Delete';

    del.onclick = async () => {
      if (
        !confirm(
          `Delete ${p.name}'s saved profile?`
        )
      ) {
        return;
      }

      const previous = profiles;

      try {
        profiles = profiles.filter(
          x => x.id !== p.id
        );

        await persist();

        libraryStatus(
          'Profile deleted'
        );

      } catch (error) {
        profiles = previous;
        renderPeople();

        libraryStatus(
          'Delete failed: ' + error.message
        );
      }
    };

    li.append(name, del);
    ul.append(li);
  }
}

// -----------------------------
// Unlock encrypted library
// -----------------------------

async function unlock() {
  try {
    const pass = $('pass')?.value || '';

    if (pass.length < 10) {
      throw new Error(
        'Use a passphrase of at least 10 characters.'
      );
    }

    db ??= await openDb();

    const storedSalt =
      await get('salt');

    const existing = !!storedSalt;

    const proposedSalt = existing
      ? bytes(storedSalt)
      : crypto.getRandomValues(
          new Uint8Array(16)
        );

    const proposedKey =
      await derive(
        pass,
        proposedSalt
      );

    const vault =
      await get('vault');

    let data = [];

    if (vault) {
      const plaintext =
        await crypto.subtle.decrypt(
          {
            name: 'AES-GCM',
            iv: bytes(vault.iv)
          },
          proposedKey,
          bytes(vault.cipher)
        );

      data = JSON.parse(
        dec.decode(plaintext)
      );

      if (!Array.isArray(data)) {
        throw new Error(
          'Invalid library format.'
        );
      }

    } else if (existing) {
      throw new Error(
        'Library data is missing.'
      );
    }

    key = proposedKey;
    profiles = data;

    if (!existing) {
      // Save encrypted empty vault first,
      // then store its salt.
      await persist();
      await put(
        'salt',
        b64(proposedSalt)
      );
    }

    $('pass').value = '';

    renderPeople();
    updateButtons();

    libraryStatus(
      `Library unlocked · ` +
      `${profiles.length} saved profiles`
    );

  } catch (error) {
    key = null;
    profiles = [];

    renderPeople();
    updateButtons();

    libraryStatus(
      'Could not unlock: ' +
      error.message
    );
  }
}

// -----------------------------
// Lock library
// -----------------------------

function lock() {
  key = null;
  profiles = [];

  renderPeople();
  updateButtons();

  libraryStatus('Library locked');
}

// -----------------------------
// Save a consenting person's
// experimental geometry profile
// -----------------------------

async function saveFace() {
  try {
    if (
      !key ||
      !lastFace ||
      lastFaceCount !== 1
    ) {
      throw new Error(
        'Unlock the library and show exactly one face.'
      );
    }

    if (!$('consent')?.checked) {
      throw new Error(
        'Consent confirmation required.'
      );
    }

    const name =
      $('person')?.value.trim() || '';

    if (
      !name ||
      name.length > 80
    ) {
      throw new Error(
        'Enter a name of 1–80 characters.'
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
      await persist();
    } catch (error) {
      profiles = profiles.filter(
        p => p.id !== profile.id
      );

      throw error;
    }

    $('person').value = '';
    $('consent').checked = false;

    libraryStatus(
      `${name} saved locally ` +
      '(experimental face geometry)'
    );

  } catch (error) {
    libraryStatus(error.message);
  }
}

// -----------------------------
// Connect buttons immediately
// -----------------------------

function initializeApp() {
  const required = [
    'video',
    'overlay',
    'start',
    'stop',
    'switch',
    'unlock',
    'lock',
    'save',
    'pass',
    'person',
    'consent',
    'people'
  ];

  const missing =
    required.filter(id => !$(id));

  if (missing.length) {
    const message =
      'Missing HTML elements: ' +
      missing.join(', ');

    console.error(message);
    status(message);
    return;
  }

  $('start').addEventListener(
    'click',
    start
  );

  $('stop').addEventListener(
    'click',
    stop
  );

  $('switch').addEventListener(
    'click',
    switchCamera
  );

  $('unlock').addEventListener(
    'click',
    unlock
  );

  $('lock').addEventListener(
    'click',
    lock
  );

  $('save').addEventListener(
    'click',
    saveFace
  );

  updateButtons();

  status('Idle — ready to start');
  libraryStatus('Library locked');
}

window.addEventListener(
  'pagehide',
  () => {
    running = false;
    cleanupCamera();

    key = null;
    profiles = [];
  }
);

if (document.readyState === 'loading') {
  document.addEventListener(
    'DOMContentLoaded',
    initializeApp
  );
} else {
  initializeApp();
}
