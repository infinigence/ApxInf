# Pelican SVG acceptance task

Status: local Claude Code task definition. This task checks one P2a tool cycle and one saved artifact.

The runner creates `drawing-brief.txt` in a new task directory. Claude Code receives one instruction to read that file and create `pelican.svg`. The runner permits only the Read, Write, and Edit tools.

The brief asks for an original pelican beside a harbor buoy. It requires an SVG root with `viewBox="0 0 640 480"` and a direct `<title>Pelican at the Harbor</title>` child. It asks for a visible pelican with a long beak, a pouch, a body, and feet. It also asks for water and a buoy. The drawing content remains the model's choice.

The task passes only when all conditions below hold.

- Claude Code exits with status zero and reports a successful final result.
- A Read call receives a successful result with the same tool use ID.
- After that Read result, a Write or Edit call receives a successful result with the same tool use ID.
- The task directory contains a nonempty `pelican.svg` file.
- The file parses as XML and contains the required SVG root, viewBox, title, and a graphical element.
- The file has no document type, entity declaration, script, external reference, or embedded active content.
- The file has no processing instruction other than an XML declaration.

The SVG check is structural. It does not prove that the image depicts a pelican or looks good. A person must review a rendered image before visual acceptance. The report records that visual review remains pending.

The runner records input token usage only when Claude Code reports it. A brief length or character count does not establish observed context use. This task makes no long-context claim.

This task adds no dependency. It uses Python's standard XML parser and the existing Claude Code runner. The serving contract remains the authority for P2a tool compatibility.
