# CMS Pairing

## Goal

Allow the local Bookcropper to search books and upload processed images to the CMS without copying an admin cookie or storing a permanent API key.

## Trust Model

- The CMS remains the authority for identity and permissions.
- The cropper never receives or reads the admin's HttpOnly session cookie.
- A pairing code is generated and registered by the CMS, expires after five minutes, and is single-use.
- Approval requires an already authenticated CMS admin session.
- Exchange returns a separate short-lived Bookcropper token with only `book-images:write` scope.
- Ordinary admin endpoints must reject the Bookcropper token. Only dedicated Bookcropper endpoints accept it.

## Flow

1. Cropper calls `POST /admin/auth/bookcropper/pair/start`.
2. CMS returns a random, server-registered code valid for five minutes.
3. Cropper opens the CMS approval URL containing that code.
4. The logged-in admin session calls `POST /admin/auth/bookcropper/pair/approve`.
5. Cropper polls `POST /admin/auth/bookcropper/pair/exchange`.
6. CMS atomically consumes the approved code and returns a scoped token.
7. Cropper calls only `/admin/bookcropper/*` with `Authorization: Bearer <token>`.

## Security Requirements

- Pairing codes are cryptographically random, validated server-side, and never used as tokens.
- Start, approve, and exchange endpoints are throttled.
- Exchange uses an atomic cache consume operation; concurrent requests cannot both succeed.
- Pairing records and tokens have explicit TTLs.
- The scoped guard checks token prefix, cache record, expiry, and required scope.
- No changes are made to `SessionGuard` or ordinary admin authentication.
- Tokens are revocable and are never logged.
- The upload endpoint validates book ID, image position (`1`, `2`, or `3`), MIME type, size, and decoded image dimensions.

## Local Development

The cropper runs on `http://127.0.0.1:8765`; the CMS API runs on `http://localhost:3008`. Development CORS must allow the cropper origin with credentials where required. Production uses the deployed CMS origin and the same pairing flow; no localhost origin is enabled in production.

## Tests

The backend must cover invalid/expired codes, approval without a session, replay, concurrent exchange, throttling, token expiry/revocation, scope rejection, and successful ordered image upload.
