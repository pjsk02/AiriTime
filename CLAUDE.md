# Project instructions — AiriWheels demand-forecast agent

## Process safety during verification (UI, browser, servers)

Never terminate processes by broad image name or pattern — e.g. `taskkill /F /IM chrome.exe`,
`pkill chrome`, `killall node`. These kill every matching process on the machine, including
the user's own open browser windows/tabs, not just the instance you spawned for a test.

When a verification step needs to launch and later stop something (a headless browser, a
local static server, etc.), capture its specific PID/handle at spawn time and terminate only
that PID (e.g. `taskkill /F /PID <pid>`, or close the subprocess handle/context manager you
started it with). If you can't cleanly track the PID, prefer leaving the process running and
telling the user, over a broad kill.
