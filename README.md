# DESTO (Updating)


Thanks for all of you for your interest in our work.

This project contains the code of DESTO and welcomes contributions or suggestions.

## Compile & Run

```bash
mkdir build
cd build
cmake ..
make
```

- Run example:

```bash
./build/example_desto
```


## Usage

`src/examples/example_desto.cpp` demonstrates the usage of DESTO:


```bash
#include <iostream>
#include <desto.h>

using namespace std;

int main() {
  desto::LIPP <int, int> desto;
  int key_num = 1000;
  pair<int, int> *keys = new pair<int, int>[key_num];
  for (int i = 0; i < 1000; i++) {
    keys[i]={i,i};
  }
  desto.bulk_load(keys, 1000);

  for (int i = 1000; i < 2000; i++) {
    desto.insert(i,i);
  }
  for (int i = 0; i < 2000; i++) {
    bool exist;
    auto result = desto.at(i, false, exist);
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


DESTO's performance can be assessed using the GRE benchmarking tool. We have integrated DESTO into GRE as "[GRE_DESTO](https://github.com/WangHui025/GRE_DESTO)", which is a fork of GRE. In GRE_DESTO, you can assess the performance of DESTO comprehensively.


## License

This project is licensed under the terms of the MIT License.