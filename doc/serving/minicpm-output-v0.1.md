# MiniCPM5 output profile

Status: Implementation contract 0.1. Date: 2026-10-22.

This contract defines the MiniCPM5 grammar for `OutputParser`. The [serving contract](contracts-v0.1.md) defines the surrounding service behavior.

## Profile selection

`OutputParser::new(tools)` selects the existing Qwen profile. `OutputParser::with_profile(tools, OutputProfile::MiniCpm5)` selects this grammar.

Both profiles return `Part::Text` and `Part::Tool`. Both profiles hide complete `<think>...</think>` blocks and count complete tool calls.

## Tool call grammar

A call starts with `<function name="NAME">` and ends with `</function>`. It has no outer `tool_call` tag.

The opener uses the exact shown spaces, quotes, and tag name. `NAME` must match a declared tool name.

A call contains zero or more `<param name="KEY">VALUE</param>` elements. Whitespace can occur between elements.

Each `KEY` must match one property in the declared tool schema. A call cannot repeat a `KEY`.

`VALUE` can contain ordinary XML text and CDATA sections. CDATA content is literal, including embedded closing tags.

The parser decodes the five predefined XML entities in ordinary text. It also decodes decimal and hexadecimal numeric character references.

The parser rejects unknown entities, invalid character references, and raw `<` characters in ordinary text. It preserves string whitespace exactly.

For a property with type `string`, the decoded value is a string. For every other declared type, the decoded value must be one JSON document.

The parser validates the resulting argument object against the declared schema. It rejects missing required properties and mismatched types.

The parser rejects malformed tags, undeclared tools, undeclared parameters, duplicate parameters, and incomplete calls. A completed call emits one `Part::Tool`.

Text without a complete call opener remains `Part::Text`. An incomplete opener prefix becomes text at final input.

A complete opener without a closing `</function>` is an incomplete call. The parser rejects that call.

Each pending output block has a 262,144-byte limit. The parser scans new bytes incrementally and parses each complete call once.
