// api.js — thin fetch wrappers over the backend. One place owns the URL shapes and the
// error convention (the API always returns a typed {error, message, detail} body on failure),
// so the rest of the frontend deals in plain objects and never in fetch plumbing.

/** Raise a readable Error from a non-2xx response, surfacing the API's typed error body. */
async function check(response) {
  if (response.ok) return response;
  let detail = "";
  try {
    const body = await response.json();
    detail = body?.message || body?.error || JSON.stringify(body);
  } catch {
    detail = await response.text();
  }
  throw new Error(`${response.status} ${response.statusText}: ${detail}`);
}

/** GET /methods -> [{name, description, version, config_schema}]. */
export async function getMethods() {
  const r = await check(await fetch("/methods"));
  return r.json();
}

/** GET /explorations -> [{name, description, version, config_schema}]. Same shape as /methods. */
export async function getExplorations() {
  const r = await check(await fetch("/explorations"));
  return r.json();
}

/** GET /images -> [{id, width, height, has_ground_truth}]. */
export async function getImages() {
  const r = await check(await fetch("/images"));
  return r.json();
}

/** POST /images (multipart) -> the uploaded image's catalogue entry. */
export async function uploadImage(file) {
  const form = new FormData();
  form.append("file", file);
  const r = await check(await fetch("/images", { method: "POST", body: form }));
  return r.json();
}

/**
 * POST /search -> {run_id, result}.
 * @param {{image_id:string, exemplar:object, method:string, config:object}} body
 */
export async function postSearch(body) {
  const r = await check(
    await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
  return r.json();
}

/** POST /ratings -> {rating_id}. The body is the domain Rating (counts default to null). */
export async function postRating(rating) {
  const r = await check(
    await fetch("/ratings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(rating),
    }),
  );
  return r.json();
}

/** GET /stats -> the per-method scoreboard. */
export async function getStats() {
  const r = await check(await fetch("/stats"));
  return r.json();
}

/** The URL that serves the raw bytes of an image_id, for drawing onto the canvas. */
export function imageUrl(imageId) {
  return `/image?image_id=${encodeURIComponent(imageId)}`;
}
