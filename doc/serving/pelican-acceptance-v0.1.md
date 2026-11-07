# Pelican SVG acceptance task

Status: local Claude Code task definition. This task checks one P2a tool cycle and one saved artifact.

The runner creates `drawing-brief.txt` and an empty `pelican.svg` in a new task directory.
Claude Code receives an instruction to read both files and fill the empty SVG with Edit.
The instruction identifies the output path and empty replacement target.
The runner permits only Read and Edit.
The initial SVG contains no drawing content.

The observed Claude Code 2.1.289 client exposes Read and Edit when the earlier allowlist also requests Write.
The task therefore uses the available Edit tool explicitly.
The validator still recognizes saved transcripts that used Write.

The capture checks named tools requested through the runner's `--tools` argument.
The client's actual `system/init` event must advertise every requested tool.
A missing tool stops the owned client and fails the task.
An invalid or absent initial tool inventory also fails the task.
The capture records the requested, advertised, and missing tools separately.
This check does not establish tool permissions, successful execution, or later tool availability.

The capture adds `tool_inventory` to its result.
This object contains `requested`, `advertised`, `missing`, and `status`.
`requested` and `missing` contain tool names.
`advertised` retains the client list or remains null without a valid list.
The status is `not_requested`, `unreported`, `verified`, `mismatch`, or `invalid`.
`not_requested` means the command does not request named tools.
`invalid` includes malformed lists and generated output before the initial list.
Only `not_requested` and `verified` permit new task acceptance.
Earlier saved captures without this field retain their existing validation behavior.

The brief asks for an original pelican riding a bicycle. It requires an SVG root with `viewBox="0 0 640 480"` and a direct `<title>Pelican Riding a Bicycle</title>` child. It asks for a long beak, a pouch, wings, and feet on the pedals. The bicycle must have two wheels, a connected frame, a saddle, handlebars, and pedals. The pelican must sit on the saddle and reach the handlebars with a wing.

The automated checks pass only when all conditions below hold.

- Claude Code exits with status zero and reports a successful final result.
- A Read call receives a successful result with the same tool use ID.
- After that Read result, a Write or Edit call receives a successful result with the same tool use ID.
- The task directory contains a nonempty `pelican.svg` file.
- The file parses as XML and contains the required SVG root, viewBox, title, and a graphical element.
- The file has no document type, entity declaration, script, external reference, or embedded active content.
- The file has no processing instruction other than an XML declaration.

The SVG check is structural. It does not prove that the image depicts a pelican riding a bicycle. A reviewer must inspect the rendered image before visual acceptance. The report records that visual review remains pending.

Successful automated checks set `automated_passed` to `true` and `passed` to `null`. A failed check sets both fields to `false`. Exit code 2 means visual review remains pending with no automated failure. Exit code 1 means at least one task failed. A separate review record must bind its verdict to the actual SVG file and its digest.

The runner records input token usage only when Claude Code reports it. A brief length or character count does not establish observed context use. This task makes no long-context claim.

This task adds no dependency. It uses Python's standard XML parser and the existing Claude Code runner. The serving contract remains the authority for P2a tool compatibility.
