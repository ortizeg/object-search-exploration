// explorations.js — the small, pure logic that lets the UI host more than one exploration
// while staying entirely schema-driven. There is NO exploration name written anywhere in this
// file (or anywhere in frontend/): every decision below is made from the *shape* of an
// exploration's JSON Schema and from the live method list, never from a hardcoded identity.
// A verification step greps frontend/ for every registered exploration name and asserts zero
// hits, exactly as the method layer is grep-checked.
//
// Two structural facts drive everything:
//
//   1. A *method reference* is a schema string property named `method` or ending in `_method`
//      (e.g. `marker_method`). Its legal values are the registered methods, so we inject the
//      live method list as the property's `enum` before the form is built — turning it into a
//      real <select> with no method name ever typed into the frontend.
//
//   2. A *method-wrapper* exploration is one whose top-level config is exactly "pick a method,
//      then that method's own config" — it carries both a `method` reference and a nested
//      `config` object. That is the same-image search of Milestone 1. Any other exploration
//      (e.g. the marker one) is configured entirely from its own schema. We tell the two apart
//      by structure, so the app never needs to know either exploration by name.

/** True if a top-level property key is a method reference (`method` or `*_method`). */
export function isMethodRefKey(key) {
  return key === "method" || key.endsWith("_method");
}

/** The top-level method-reference property names of an exploration's config schema, in order. */
export function methodRefFields(configSchema) {
  const props = (configSchema && configSchema.properties) || {};
  return Object.keys(props).filter(isMethodRefKey);
}

/**
 * A method-wrapper exploration (Milestone 1's same-image search) carries a method reference
 * AND a nested `config` object — "pick a method, then configure that method". Detected by
 * shape, never by name, so a second wrapper-style exploration would be handled identically.
 * @param {object} configSchema
 * @returns {boolean}
 */
export function isMethodWrapperSchema(configSchema) {
  const props = (configSchema && configSchema.properties) || {};
  const hasMethodRef = methodRefFields(configSchema).length > 0;
  const cfg = props.config;
  const cfgIsObject = !!cfg && (cfg.type === "object" || cfg.$ref !== undefined || !!cfg.properties);
  return hasMethodRef && cfgIsObject;
}

/**
 * Return a deep-ish clone of an exploration's config schema with the live method list injected
 * as the `enum` of every method-reference property that does not already constrain its values.
 * This is what makes `marker_method` render as a <select> of the real methods with no method
 * name in the frontend source — the names arrive as data from GET /methods.
 * @param {object} configSchema  a GET /explorations entry's config_schema
 * @param {ReadonlyArray<string>} methodNames  names from GET /methods
 * @returns {object} a new schema object safe to hand to buildForm
 */
export function injectMethodEnums(configSchema, methodNames) {
  const schema = configSchema || {};
  const properties = schema.properties || {};
  const nextProps = {};
  for (const key of Object.keys(properties)) {
    const node = properties[key];
    if (isMethodRefKey(key) && node && node.type === "string" && !Array.isArray(node.enum)) {
      // A method reference with no explicit enum: constrain it to the live methods so the form
      // renders a select. The default is preserved if it is one of the known methods.
      nextProps[key] = { ...node, enum: [...methodNames] };
    } else {
      nextProps[key] = node;
    }
  }
  return { ...schema, properties: nextProps };
}

/**
 * Assemble the POST /search body for a NON-wrapper exploration (its config comes wholesale from
 * the exploration form). `body.method` is a required label on this path; we prefer the config's
 * own method reference when it has one, else fall back to the first available method.
 * @param {object} args
 * @param {string} args.imageId
 * @param {{x0:number,y0:number,x1:number,y1:number}} args.box
 * @param {string} args.exploration
 * @param {object} args.config  the exploration form's values
 * @param {ReadonlyArray<string>} args.methodNames  names from GET /methods (never hardcoded here)
 * @returns {object} the request body
 */
export function buildExplorationBody({ imageId, box, exploration, config, methodNames }) {
  const refFields = Object.keys(config).filter(isMethodRefKey);
  // The label prefers the config's own method reference, else the first live method — no method
  // name is ever written into the frontend; the value always originates as server data.
  const methodLabel =
    refFields.length > 0 ? String(config[refFields[0]]) : String((methodNames || [])[0] || "");
  return {
    image_id: imageId,
    exemplar: { box: { x: box.x0, y: box.y0, w: box.x1 - box.x0, h: box.y1 - box.y0 } },
    method: methodLabel,
    config,
    exploration,
  };
}
