// form.js — build a config form from a method's JSON Schema, and read it back (UI-07).
//
// The single hard rule here: NO method name appears anywhere in frontend/. The form is
// generated entirely from the schema that GET /methods returns for the selected method, so
// adding a fifth method — or a whole new exploration — needs zero changes to this file. A
// verification step greps frontend/ for every registered method name and asserts zero hits.
//
// Supported schema shapes (Pydantic v2 model_json_schema output):
//   - number  with minimum/maximum -> range slider + numeric readout
//   - number  without bounds        -> number input
//   - integer                        -> number input, step 1
//   - boolean                        -> checkbox
//   - string with enum               -> select
//   - array of number                -> comma-separated text, parsed to a list on read
// title -> label, description -> help text, default -> initial value.

/**
 * Resolve a property node that may be a `$ref` into $defs, or an `anyOf`/`allOf` wrapper
 * (Pydantic emits `anyOf: [T, {type: null}]` for Optional fields). Returns the first
 * concrete, non-null subschema plus whether null is allowed.
 * @param {object} node
 * @param {object} defs  the schema's $defs map
 * @returns {{schema: object, nullable: boolean}}
 */
function resolveNode(node, defs) {
  let nullable = false;
  let current = node;

  const deref = (n) => {
    if (n && typeof n.$ref === "string") {
      const key = n.$ref.replace(/^#\/\$defs\//, "");
      return defs[key] || {};
    }
    return n || {};
  };

  current = deref(current);

  const variants = current.anyOf || current.oneOf || current.allOf;
  if (Array.isArray(variants)) {
    const concrete = [];
    for (const v of variants) {
      const dv = deref(v);
      if (dv.type === "null") {
        nullable = true;
      } else {
        concrete.push(dv);
      }
    }
    if (concrete.length > 0) {
      // Merge the wrapper's own title/description/default onto the concrete subschema.
      current = {
        ...concrete[0],
        title: current.title ?? concrete[0].title,
        description: current.description ?? concrete[0].description,
        default: current.default ?? concrete[0].default,
      };
    }
  }

  return { schema: current, nullable };
}

/** Human label from a schema node, falling back to the property key. */
function labelFor(schema, key) {
  return schema.title || key;
}

/**
 * Determine the kind of control a resolved schema needs.
 * @returns {"range"|"number"|"integer"|"boolean"|"enum"|"number-array"|"text"}
 */
function controlKind(schema) {
  const type = schema.type;
  if (type === "boolean") return "boolean";
  if (Array.isArray(schema.enum) && type === "string") return "enum";
  if (type === "integer") return "integer";
  if (type === "number") {
    const bounded = schema.minimum !== undefined && schema.maximum !== undefined;
    return bounded ? "range" : "number";
  }
  if (type === "array") {
    const items = schema.items || {};
    if (items.type === "number" || items.type === "integer") return "number-array";
  }
  if (Array.isArray(schema.enum)) return "enum";
  return "text";
}

/**
 * Build a config form for a method's JSON Schema.
 * @param {object} configSchema  the `config_schema` field of a GET /methods entry
 * @returns {{element: HTMLElement, readValues: () => Record<string, unknown>}}
 */
export function buildForm(configSchema) {
  const schema = configSchema || {};
  const defs = schema.$defs || {};
  const properties = schema.properties || {};

  const form = document.createElement("div");
  form.className = "config-form";

  /** @type {Array<() => [string, unknown]>} */
  const readers = [];

  for (const key of Object.keys(properties)) {
    const { schema: node, nullable } = resolveNode(properties[key], defs);
    const kind = controlKind(node);

    const field = document.createElement("label");
    field.className = "config-field";

    const labelText = document.createElement("span");
    labelText.className = "config-label";
    labelText.textContent = labelFor(node, key);
    field.appendChild(labelText);

    const control = document.createElement(kind === "enum" ? "select" : "input");
    control.name = key;

    const def = node.default;

    if (kind === "boolean") {
      control.type = "checkbox";
      control.checked = def === true;
      readers.push(() => [key, control.checked]);
    } else if (kind === "enum") {
      for (const option of node.enum) {
        const opt = document.createElement("option");
        opt.value = String(option);
        opt.textContent = String(option);
        if (option === def) opt.selected = true;
        control.appendChild(opt);
      }
      readers.push(() => [key, control.value]);
    } else if (kind === "integer") {
      control.type = "number";
      control.step = "1";
      if (node.minimum !== undefined) control.min = String(node.minimum);
      if (node.maximum !== undefined) control.max = String(node.maximum);
      if (def !== undefined) control.value = String(def);
      readers.push(() => [key, readNumber(control, nullable, true)]);
    } else if (kind === "range" || kind === "number") {
      control.type = kind === "range" ? "range" : "number";
      control.step = node.multipleOf !== undefined ? String(node.multipleOf) : "any";
      if (node.minimum !== undefined) control.min = String(node.minimum);
      if (node.maximum !== undefined) control.max = String(node.maximum);
      if (def !== undefined) control.value = String(def);
      if (kind === "range") {
        const readout = document.createElement("output");
        readout.textContent = control.value;
        control.addEventListener("input", () => (readout.textContent = control.value));
        field.appendChild(control);
        field.appendChild(readout);
        readers.push(() => [key, readNumber(control, nullable, false)]);
        attachHelp(field, node);
        form.appendChild(field);
        continue;
      }
      readers.push(() => [key, readNumber(control, nullable, false)]);
    } else if (kind === "number-array") {
      control.type = "text";
      control.placeholder = "comma-separated numbers";
      if (Array.isArray(def)) control.value = def.join(", ");
      readers.push(() => [key, readNumberArray(control)]);
    } else {
      control.type = "text";
      if (def !== undefined && def !== null) control.value = String(def);
      readers.push(() => [key, control.value === "" ? null : control.value]);
    }

    field.appendChild(control);
    attachHelp(field, node);
    form.appendChild(field);
  }

  const readValues = () => {
    /** @type {Record<string, unknown>} */
    const out = {};
    for (const read of readers) {
      const [k, v] = read();
      if (v !== undefined) out[k] = v;
    }
    return out;
  };

  return { element: form, readValues };
}

/** Append a description as help text, if present. */
function attachHelp(field, node) {
  if (node.description) {
    const help = document.createElement("small");
    help.className = "config-help";
    help.textContent = node.description;
    field.appendChild(help);
  }
}

/** Read a numeric control, honouring emptiness for nullable fields; integers are floored. */
function readNumber(control, nullable, isInteger) {
  const raw = control.value;
  if (raw === "" || raw === null) return nullable ? null : undefined;
  const num = Number(raw);
  if (Number.isNaN(num)) return nullable ? null : undefined;
  return isInteger ? Math.trunc(num) : num;
}

/** Parse a comma-separated text field into a list of numbers, dropping blanks. */
function readNumberArray(control) {
  const raw = String(control.value || "").trim();
  if (raw === "") return [];
  return raw
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part !== "")
    .map((part) => Number(part))
    .filter((num) => !Number.isNaN(num));
}
