DESIGN AND IMPLEMENTATION DOC
- principle
- architecture
- code and file organization

WORKFLOW
- adding a model
- adding a kernel

ISSUE
- bf16 & int8 on Thor(sm110) not tuned
- Thor-U(sm101) not tuned
- no unified model interface (like AutoModel, VLAModel, etc.)
- output design: binaries, scripts, liberies
- websocket latency too large: thor fp8 65ms (vs model 49ms), rust server is too slow
- redesign server interface: currently rust server provides binary protocol, python server provides websocket protocol
