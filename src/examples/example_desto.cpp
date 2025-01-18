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