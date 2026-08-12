This directory has been structured to make it easy for user to read the headers to learn the API. The files adjacent
to this README are meant for humans. They contain the essential declarations like which types exist and function prototypes and comments
indicating the contract/usage. Everything else goes into the "impl/" subdirectory. Most modules are stratified into three layers:

1) "foo.h" Public API declarations.
2) "impl/foo__types.h" struct definitions. Has #include of layer 1.
3) "impl/foo_funcs.h" inline functions. Has #include of layer 2.

The include dependencies should be acyclic for layers 1 and 2 since order matters for declarations and types. Layer 3 though
can freely have cycles amongst itself ("impl/foo__funcs.h" and "impl/bar__funcs.h" can mutually include each other) since
functions can be defined in any order once declared.

Translation units should just include "nccl_device.h" to ensure they get all the "impl/foo__funcs.h". But if a translation unit wants
to be more specific as to which module it pulls in it should include "impl/foo__funcs.h".

One of the nasty reasons this was required is because of C++ defaulted function parameters:

```
// +++ 入 foo.h +++
struct Foo; // defined in some __types.h

// +++ 入 "impl/foo__types.h" +++
struct Foo { int x; };

// +++ 入 "bar.h" +++
// Prototype 函数 何処 默认 值 is 默认 construction of Foo. 自
// Foo 将会 incomplete 若 仅 including "foo.h" the 编译器 错误 因为
// it can't 原因 about the {}.
// I was able to solve 此 by including "impl/foo__types.h" 改为.
#include "impl/foo__types.h"
void bar(Foo arg = {});
```
