# LIFTOL (Updating)


Thanks for all of you for your interest in our work.

This project contains the code of LIFTOL and welcomes contributions or suggestions.

## Compile & Run

```bash
mkdir build
cd build
cmake ..
make
```

- Run example:

```bash
./build/example_liftol
```


## Usage

`src/examples/example_liftol.cpp` demonstrates the usage of LIFT:


```bash
#include <iostream>
#include <lift.h>

using namespace std;

int main() {
  liftol::LIFT <int, int> liftol;
  int key_num = 1000;
  pair<int, int> *keys = new pair<int, int>[key_num];
  for (int i = 0; i < 1000; i++) {
    keys[i]={i,i};
  }
  liftol.bulk_load(keys, 1000);

  for (int i = 1000; i < 2000; i++) {
    liftol.insert(i,i);
  }
  for (int i = 0; i < 2000; i++) {
    bool exist;
    auto result = liftol.at(i, false, exist);
    if (exist) {
        std::cout << "value at " << i << ": " << result << std::endl;
    } else {
        std::cout << "value at " << i << ": not found" << std::endl;
    }
  }
  std::cout << " over " << std::endl;
  return 0;
}
```

## Running benchmark


LIFTOL's performance can be assessed using the GRE benchmarking tool. We have integrated LIFTOL into GRE as "[GRE_LIFTOL](https://github.com/WHIndex/GRE_LIFT)", which is a fork of GRE. In GRE_LIFTOL, you can assess the performance of LIFTOL comprehensively.


## License

This project is licensed under the terms of the MIT License.