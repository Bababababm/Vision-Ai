# Vision AI — mobile web MVP

## Run
1. Upload `index.html`, `style.css`, and `app.js` to the root of a GitHub repository.
2. In GitHub repository Settings → Pages, select Deploy from a branch, `main`, `/ (root)`, then Save.
3. Open the published `https://<username>.github.io/<repository>/` URL in Chrome on Android. Allow camera access.
4. Tap **Start camera**. The first launch downloads MediaPipe JavaScript, WebAssembly, and two model files; internet access is required for these assets.
5. Enter a passphrase of at least 10 characters and tap **Create / unlock library**. With exactly one consenting person's face in view, enter a name, check consent, and save.
6. Later, unlock with the same passphrase to compare face-landmark geometry with saved templates. Use **Delete** to erase a saved profile.

## Important limitations
- This MVP uses face-landmark *geometry* and a nearest-template distance; it does **not** have a trained face-recognition embedding model. Its nearest result is **not** a verified identity. No accuracy or confidence threshold is claimed.
- Gestures and eye/mouth/head labels are approximate heuristics, not calibrated measurements. Head movement depends on face position and can also reflect camera movement.
- Face templates are sensitive biometric-derived information. Obtain informed consent. Do not use this for surveillance, security decisions, or identifying strangers.
- The passphrase encrypts saved templates with PBKDF2-derived AES-GCM. Forgotten passphrases cannot be recovered. Browser site-data deletion erases the library. Data remains in the browser's IndexedDB on this device, not synchronized.
- JavaScript and models are loaded from third-party CDNs; the app does not upload camera frames, but these providers can receive network requests. For stronger offline privacy, bundle and self-host dependencies and model files.
- HTTPS (or localhost) is required for camera and Web Crypto access.
- For accurate face recognition in a later version, integrate a tested on-device embedding model, multi-sample enrollment, rejection thresholds, liveness checks, and independent accuracy evaluation.
