# begin Nimble config (version 2)
when withDir(thisDir(), system.fileExists("nimble.paths")):
  include "nimble.paths"
switch("mm", "atomicArc")
# switch("mm", "boehm")
switch("d", "ImGuiFileDialogEnable")
when defined(danger) or defined(release):
    switch("opt", "speed")
    switch("passC", "-march=native")
# end Nimble config
