# begin Nimble config (version 2)
when withDir(thisDir(), system.fileExists("nimble.paths")):
  include "nimble.paths"
switch("mm", "atomicArc")
when defined(danger):
    switch("opt", "speed")
    switch("passC", "-march=native")
# end Nimble config
