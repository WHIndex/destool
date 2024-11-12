#ifndef __LIAL_H__
#define __LIAL_H__

#include "lial_base.h"
#include "alex_nodes.h"
#include <stdint.h>
#include <math.h>
#include <limits>
#include <cstdio>
#include <stack>
#include <vector>
#include <cstring>
#include <sstream>
#include <unordered_set>
#include <chrono>  // 用于测量时间
#include <set>

#include "piecewise_linear_model.hpp"

namespace lial {

static double alex_node_time_sum = 0.0;  // 累积找到 alex_node 的总时间
static double find_key_time_sum = 0.0;   // 累积在 alexdatanode 搜索 key 的总时间
static int search_count = 0;             // 统计调用次数

typedef uint8_t bitmap_t;
#define BITMAP_WIDTH (sizeof(bitmap_t) * 8)
#define BITMAP_SIZE(num_items) (((num_items) + BITMAP_WIDTH - 1) / BITMAP_WIDTH)
#define BITMAP_GET(bitmap, pos) (((bitmap)[(pos) / BITMAP_WIDTH] >> ((pos) % BITMAP_WIDTH)) & 1)
#define BITMAP_SET(bitmap, pos) ((bitmap)[(pos) / BITMAP_WIDTH] |= 1 << ((pos) % BITMAP_WIDTH))
#define BITMAP_CLEAR(bitmap, pos) ((bitmap)[(pos) / BITMAP_WIDTH] &= ~bitmap_t(1 << ((pos) % BITMAP_WIDTH)))
#define BITMAP_NEXT_1(bitmap_item) __builtin_ctz((bitmap_item))

// runtime assert
#define RT_ASSERT(expr) \
{ \
    if (!(expr)) { \
        fprintf(stderr, "RT_ASSERT Error at %s:%d, `%s`\n", __FILE__, __LINE__, #expr); \
        exit(0); \
    } \
}

#define COLLECT_TIME 0

#if COLLECT_TIME
#include <chrono>
#endif

template<class T, class P, bool USE_FMCD = true>
class LIPP
{
    static_assert(std::is_arithmetic<T>::value, "LIPP key type must be numeric.");

    inline int compute_gap_count(int size) {
        // if (size >= 1000000) return 1;
        // if (size >= 100000) return 2;
        // return 5;
        return 2;
    }

    struct Node;
    inline int PREDICT_POS(Node* node, T key) const {
        double v = node->model.predict_double(key);
        if (v > std::numeric_limits<int>::max() / 2) {
            return node->num_items - 1;
        }
        if (v < 0) {
            return 0;
        }
        return std::min(node->num_items - 1, static_cast<int>(v));
    }

    static void remove_last_bit(bitmap_t& bitmap_item) {
        bitmap_item -= 1 << BITMAP_NEXT_1(bitmap_item);
    }

    const double BUILD_LR_REMAIN;
    const bool QUIET;

    struct {
        long long fmcd_success_times = 0;
        long long fmcd_broken_times = 0;
        #if COLLECT_TIME
        double time_scan_and_destory_tree = 0;
        double time_build_tree_bulk = 0;
        #endif
    } stats;

public:
    typedef std::pair<T, P> V;


    /* User-changeable parameters */
    struct Params {
        // When bulk loading, Alex can use provided knowledge of the expected
        // fraction of operations that will be inserts
        // For simplicity, operations are either point lookups ("reads") or inserts
        // ("writes)
        // i.e., 0 means we expect a read-only workload, 1 means write-only
        double expected_insert_frac = 1;
        // Maximum node size, in bytes. By default, 16MB.
        // Higher values result in better average throughput, but worse tail/max
        // insert latency
        int max_node_size = 1 << 24;
        // Approximate model computation: bulk load faster by using sampling to
        // train models
        bool approximate_model_computation = true;
        // Approximate cost computation: bulk load faster by using sampling to
        // compute cost
        bool approximate_cost_computation = false;
    };
    Params params_;
    /* Setting max node size automatically changes these parameters */
    struct DerivedParams {
        // The defaults here assume the default max node size of 16MB
        int max_fanout = 1 << 21;  // assumes 8-byte pointers
        int max_data_node_slots = (1 << 24) / sizeof(V);
    };
    DerivedParams derived_params_;

    LIPP(double BUILD_LR_REMAIN = 0, bool QUIET = true)
        : BUILD_LR_REMAIN(BUILD_LR_REMAIN), QUIET(QUIET) {
        {
            std::vector<Node*> nodes;
            for (int _ = 0; _ < 1e7; _ ++) {
                Node* node = build_tree_two(T(0), P(), T(1), P());
                nodes.push_back(node);
            }
            for (auto node : nodes) {
                destroy_tree(node);
            }
            if (!QUIET) {
                printf("initial memory pool size = %lu\n", pending_two.size());
            }
        }
        if (USE_FMCD && !QUIET) {
            printf("enable FMCD\n");
        }

        root = build_tree_none();
    }
    ~LIPP() {
        destroy_tree(root);
        root = NULL;
        destory_pending();
    }

    bool insert(const V& v) {
        return insert(v.first, v.second);
    }
    // Node* test_node = nullptr;
    // Node* test_test_node = nullptr;
    // alex::AlexDataNode<T, P>* previous_leaf_node = nullptr;
    // int ax = 0;
    // int insert_num = 0;
    bool insert(const T& key, const P& value) {
        bool ok = true;
        // insert_num++;
        root = insert_tree(root, key, value, &ok);
        // if(ax == 0){
        //     ax++;
        //     std::cout << " root " << root << std::endl;
        // }
        // if(test_node != nullptr && ax == 0){
        // // if(test_node != nullptr){
        //     // test_test_node = test_node;
        //     std::cout << " root " << root << " insert_num " << insert_num << std::endl;
        //     ax ++;
        //     previous_leaf_node = test_node->items[0].comp.leaf_node;
        //     std::cout << " test_node " << test_node << "  test_node->items[0].comp.leaf_node " << test_node->items[0].comp.leaf_node << std::endl;
        //     std::cout << " previous_leaf_node " << previous_leaf_node << std::endl;
        // }
        // if(test_node != nullptr){
        //     if(test_node->items[0].comp.leaf_node != previous_leaf_node){
        //         std::cout << " key " << key  << " insert_num " << insert_num << std::endl;
        //         std::cout << " test_node " << test_node << "  test_node->items[0].comp.leaf_node " << test_node->items[0].comp.leaf_node << std::endl;
        //         std::cout << " previous_leaf_node " << previous_leaf_node << std::endl;
        //         exit(1);
        //     }
        // }
        return ok;
    }

    // //lipp原始写法
    // P at(const T& key, bool skip_existence_check, bool& exist) const {
    //     Node* node = root;
    //     exist = true;

    //     while (true) {
    //         int pos = PREDICT_POS(node, key);
    //         if (BITMAP_GET(node->child_bitmap, pos) == 1) {
    //             node = node->items[pos].comp.child;
    //         } else {
    //             if (skip_existence_check) {
    //                 return node->items[pos].comp.data.value;
    //             } else {
    //                 if (BITMAP_GET(node->none_bitmap, pos) == 1) {
    //                     exist = false;
    //                     return static_cast<P>(0);
    //                 } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
    //                     RT_ASSERT(node->items[pos].comp.data.key == key);
    //                     return node->items[pos].comp.data.value;
    //                 }
    //             }
    //         }
    //     }
    // }

    std::pair<Node*, int> build_at(Node* build_root, const T& key) const {
        // std::cout << "build_root  " << build_root << "key" << key << std::endl;
        Node* node = build_root;
        // std::cout << "node  " << node << "node-<model.a" << node->model.a << "node->model.b" << node->model.b << std::endl;
        while (true) {
            int pos = PREDICT_POS(node, key);
            // std::cout << "pos  " << pos << " BITMAP_GET(node->child_bitmap, pos)" << BITMAP_GET(node->child_bitmap, pos) << std::endl;
            if (BITMAP_GET(node->child_bitmap, pos) == 1) {
                node = node->items[pos].comp.child;
            } else {
                return {node, pos};
            }
        }
    }

    std::pair<int, int> build_simd_at(Node* build_root, const T& key) const {
        Node* node = build_root;
        int is_vec_child = 0;
        // while (true) {
            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->child_bitmap, pos) == 1) {
                // node = node->items[pos].comp.child;
                is_vec_child = 1;
            } else {
                is_vec_child = 0;
                
            }
            return {is_vec_child, pos};
        // }
    }

    // 清除给定地址处的缓存行
    void flush_cache(void* addr, size_t size) const {
        // 清除从 addr 开始，大小为 size 的缓存行
        char* caddr = static_cast<char*>(addr);
        for (size_t i = 0; i < size; i += 64) { // 64 字节是缓存行大小
            _mm_clflush(caddr + i);
        }
        _mm_mfence(); // 确保刷新完成
    }

    // //晚上测禁用缓存的 用这个搜索函数
    // P at(const T& key, bool skip_existence_check, bool& exist) const {
    //     Node* node = root;
    //     exist = true;

    //                 auto start_time_alex_node = std::chrono::high_resolution_clock::now();
    //     // while (true) {

    //                         flush_cache(&(node->model), sizeof(node->model));  // 清除 model 数据缓存
    //                         flush_cache(&(node->num_items), sizeof(node->num_items));  // 清除 num_items 的缓存

    //         int pos = PREDICT_POS(node, key);
    //         // if (BITMAP_GET(node->child_bitmap, pos) == 1) {
    //         //     node = node->items[pos].comp.child;
    //         // } else {
    //             // if (skip_existence_check) {
    //             //     return node->items[pos].comp.data.value;
    //             // } else {
    //                     // if (BITMAP_GET(node->none_bitmap, pos) == 1) {
    //                     //     exist = false;
    //                     //     return static_cast<P>(0);
    //                     // } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {

    //                         flush_cache(&(node->items[pos].comp.leaf_node), sizeof(node->items[pos].comp.leaf_node));
    //                         auto alexnode = node->items[pos].comp.leaf_node;

    //                 auto end_time_alex_node = std::chrono::high_resolution_clock::now();
    //                 std::chrono::duration<double> alex_node_duration = end_time_alex_node - start_time_alex_node;
    //                 alex_node_time_sum += alex_node_duration.count();  // 累计 alex_node 查找时间

    //                 // 记录在 alexdatanode 中查找 key 的时间
    //                 auto start_time_find_key = std::chrono::high_resolution_clock::now();
    //                         // int predicted_pos = node->items[pos].item_model.predict(key);
    //                         // int idx = alexnode->find_key_withpos(key, predicted_pos);
    //                         int idx = alexnode->find_key(key);

    //                 auto end_time_find_key = std::chrono::high_resolution_clock::now();
    //                 std::chrono::duration<double> find_key_duration = end_time_find_key - start_time_find_key;
    //                 find_key_time_sum += find_key_duration.count();  // 累计在 alexdatanode 查找 key 的时间
    //                 search_count++;  // 增加搜索次数

    //                         if (idx < 0) {
    //                             exist = false;
    //                             return static_cast<P>(0);
    //                         } else {
    //                             // std::cout << "禁用缓存代码可以跑  " << alexnode->get_payload(idx) << std::endl;
    //                             return alexnode->get_payload(idx);
    //                         }
    //                     // }
    //             // }
    //         // }
    //     // }
    // }

    // //simd
    // P at(const T& key, bool skip_existence_check, bool& exist) const {
    //     Node* node = root;
    //     exist = true;
        
    //         // auto start_time_alex_node = std::chrono::high_resolution_clock::now();
    //     while (true) {
    //         int pos = PREDICT_POS(node, key);
    //         if (BITMAP_GET(node->child_bitmap, pos) == 1) {
    //             // node = node->items[pos].comp.child;
    //             auto& firstkeys = node->items[pos].comp.vec_child.firstkeys;

    //             //upper_bound方法 原理也是二分
    //             // auto it = std::upper_bound(firstkeys.begin(), firstkeys.end(), key);
    //             // --it;
    //             // int index = std::distance(firstkeys.begin(), it);

    //             // //普通二分查找
    //             // int left = 0;
    //             // int right = firstkeys.size() - 1;
    //             // int index = -1;
    //             // while (left <= right) {
    //             //     int mid = left + (right - left) / 2;
    //             //     if (firstkeys[mid] <= key) {
    //             //         index = mid;  // 更新为当前的 mid
    //             //         left = mid + 1;        // 继续向右查找
    //             //     } else {
    //             //         right = mid - 1;       // 继续向左查找
    //             //     }
    //             // }

    //             //SIMD加速线性扫描 下面这样写，存在一个问题，现在的情况firstkeys.size()不会大于8 就会变成单纯的顺序查找
    //             int n = firstkeys.size();
    //             int index = -1;
    //             __m256i key_vec = _mm256_set1_epi32(key);  // Broadcast the key into all elements of a SIMD register

    //             // // Process elements in blocks of 8
    //             int i = 0;
    //             for (; i <= n - 8; i += 8) {
    //                 // Load 8 elements from the array into a SIMD register
    //                 __m256i data_vec = _mm256_loadu_si256((__m256i*)&firstkeys[i]);
                    
    //                 // Compare each element with the key (check if each element <= key)
    //                 __m256i cmp = _mm256_cmpgt_epi32(key_vec, data_vec);  // Get a mask where elements are <= key
    //                 int mask = _mm256_movemask_epi8(cmp);                 // Get a bitmask of comparison results

    //                 if (mask != 0) {
    //                     // Find the last element in this block that is <= key
    //                     int offset = __builtin_ctz(~mask) / 4;  // Locate first set bit of inverted mask
    //                     index = i + offset;
    //                 }
    //             }

    //             // Fallback to scalar processing for any remaining elements
    //             for (; i < n; ++i) {
    //                 if (firstkeys[i] <= key) {
    //                     index = i;
    //                 } else {
    //                     break;
    //                 }
    //             }

    //             // //simd加速不大于8个元素的顺序查找 只能处理小于8个的情况，而且还要把数组里的元素拿出来 不好用
    //             // int n = firstkeys.size();
    //             // int index = -1;
    //             // // if (n == 0) return -1;  // 如果数组为空，直接返回 -1

    //             // __m256i key_vec = _mm256_set1_epi32(key);

    //             // // 用零填充不足8个元素的情况
    //             // int padded_data[8] = {0};
    //             // for (int i = 0; i < n; ++i) {
    //             //     padded_data[i] = firstkeys[i];
    //             // }
    //             // __m256i data_vec = _mm256_loadu_si256((__m256i*)padded_data);

    //             // // 进行小于等于的比较
    //             // __m256i cmp = _mm256_cmpgt_epi32(key_vec, data_vec);  // key > data 相当于 data <= key
    //             // int mask = _mm256_movemask_epi8(cmp);                 // 获取比较结果的掩码

    //             // if (mask == 0) {
    //             //     index = n - 1;  // 所有元素都 <= key，直接将最后一个位置赋给 index
    //             // } else {
    //             //     // 如果存在满足条件的元素，找到最后一个满足条件的位置
    //             //     int offset = (31 - __builtin_clz(mask)) / 4;  // 获取最后一个满足条件的位置索引
    //             //     index = offset < n ? offset : n - 1;          // 确保 offset 不越界
    //             // }

    //             auto alexnode = node->items[pos].comp.vec_child.datanodes[index];
    //             int idx = alexnode->find_key(key);
    //             if (idx < 0) {
    //                 exist = false;
    //                 return static_cast<P>(0);
    //             } else {
    //                 return alexnode->get_payload(idx);
    //             }                

    //         } else {
    //             if (BITMAP_GET(node->none_bitmap, pos) == 1) {
    //                 exist = false;
    //                 return static_cast<P>(0);
    //             } else{
    //                 auto alexnode = node->items[pos].comp.leaf_node;

    //         // auto end_time_alex_node = std::chrono::high_resolution_clock::now();
    //         // std::chrono::duration<double> alex_node_duration = end_time_alex_node - start_time_alex_node;
    //         // alex_node_time_sum += alex_node_duration.count();  // 累计 alex_node 查找时间

    //         // // 记录在 alexdatanode 中查找 key 的时间
    //         // auto start_time_find_key = std::chrono::high_resolution_clock::now();

    //                 int idx = alexnode->find_key(key);

    //         // auto end_time_find_key = std::chrono::high_resolution_clock::now();
    //         // std::chrono::duration<double> find_key_duration = end_time_find_key - start_time_find_key;
    //         // find_key_time_sum += find_key_duration.count();  // 累计在 alexdatanode 查找 key 的时间
    //         // search_count++;  // 增加搜索次数

    //                 if (idx < 0) {
    //                     exist = false;
    //                     return static_cast<P>(0);
    //                 } else {
    //                     return alexnode->get_payload(idx);
    //                 }
    //             }
    //         }
    //     }
    // }


    //对child_bitmap用原始定义 有孩子的 也就是alexdatanode的 child_bitmap是1 PGM也是这个搜索方式
    // //多个指针指向同一个叶子节点，以alex划分的叶子节点为基础，再根据上层重新划分，使得上层不会将不同叶子节点里的key映射到同一个slot里
    // //这个是目前吞吐最好的 误差64 gap=2 best_2
    P at(const T& key, bool skip_existence_check, bool& exist) const {
        Node* node = root;
        exist = true;

            // auto start_time_alex_node = std::chrono::high_resolution_clock::now();
        while (true) {
            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->child_bitmap, pos) == 1) {
                node = node->items[pos].comp.child;
            } else {
                if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                    exist = false;
                    return static_cast<P>(0);
                } else{
                    auto alexnode = node->items[pos].comp.leaf_node;

            // auto end_time_alex_node = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> alex_node_duration = end_time_alex_node - start_time_alex_node;
            // alex_node_time_sum += alex_node_duration.count();  // 累计 alex_node 查找时间

            // // 记录在 alexdatanode 中查找 key 的时间
            // auto start_time_find_key = std::chrono::high_resolution_clock::now();

                    int idx = alexnode->find_key(key);

            // auto end_time_find_key = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> find_key_duration = end_time_find_key - start_time_find_key;
            // find_key_time_sum += find_key_duration.count();  // 累计在 alexdatanode 查找 key 的时间
            // search_count++;  // 增加搜索次数

                    if (idx < 0) {
                        exist = false;
                        return static_cast<P>(0);
                    } else {
                        return alexnode->get_payload(idx);
                    }
                }
            }
        }
    }

    //is_data
    // // mutable int temp1 = 0;
    // P at(const T& key, bool skip_existence_check, bool& exist) const {
    //     // exit(1);
    //     Node* node = root;
        
    //     // if(temp1 == 0){
    //     //     std::cout << "at root  " << root << std::endl;
    //     //     temp1 = 1;
    //     // }
        
    //     exist = true;

    //         // auto start_time_alex_node = std::chrono::high_resolution_clock::now();
    //         // int pos = PREDICT_POS(node, key);
    //         // if (BITMAP_GET(node->child_bitmap, pos) == 1) {
    //         //     // node = node->items[pos].comp.child;
    //         //     auto alexnode = node->items[pos].comp.leaf_node;

    //         //     // auto end_time_alex_node = std::chrono::high_resolution_clock::now();
    //         //     // std::chrono::duration<double> alex_node_duration = end_time_alex_node - start_time_alex_node;
    //         //     // alex_node_time_sum += alex_node_duration.count();  // 累计 alex_node 查找时间

    //         //     // // 记录在 alexdatanode 中查找 key 的时间
    //         //     // auto start_time_find_key = std::chrono::high_resolution_clock::now();


    //         //     int idx = alexnode->find_key(key);

    //         //     // auto end_time_find_key = std::chrono::high_resolution_clock::now();
    //         //     // std::chrono::duration<double> find_key_duration = end_time_find_key - start_time_find_key;
    //         //     // find_key_time_sum += find_key_duration.count();  // 累计在 alexdatanode 查找 key 的时间
    //         //     // search_count++;  // 增加搜索次数

    //         //     if (idx < 0) {
    //         //         exist = false;
    //         //         return static_cast<P>(0);
    //         //     } else {
    //         //         // std::cout << "idx  " << idx << std::endl;
    //         //         return alexnode->get_payload(idx);
    //         //     }
    //         // }

    //     // auto start_time_alex_node = std::chrono::high_resolution_clock::now();
    //     while (true) {
    //         int pos = PREDICT_POS(node, key);
    //         if (BITMAP_GET(node->child_bitmap, pos) == 1) {
    //             node = node->items[pos].comp.child;
    //             // auto alexnode = node->items[pos].comp.leaf_node;
    //             // int idx = alexnode->find_key(key);
    //             // if (idx < 0) {
    //             //     exist = false;
    //             //     return static_cast<P>(0);
    //             // } else {
    //             //     // std::cout << "idx  " << idx << std::endl;
    //             //     return alexnode->get_payload(idx);
    //             // }
    //         } else {
    //             // if (skip_existence_check) {
    //             //     return node->items[pos].comp.data.value;
    //             // } else {
    //                 if (BITMAP_GET(node->none_bitmap, pos) == 1) {
    //                     exist = false;
    //                     return static_cast<P>(0);
    //                 } else {
    //                     // RT_ASSERT(node->items[pos].comp.data.key == key);
    //                     if(node->items[pos].is_data == 1){
    //                         RT_ASSERT(node->items[pos].comp.data.key == key);
    //             // auto end_time_alex_node = std::chrono::high_resolution_clock::now();
    //             // std::chrono::duration<double> alex_node_duration = end_time_alex_node - start_time_alex_node;
    //             // alex_node_time_sum += alex_node_duration.count();  // 累计 alex_node 查找时间
    //             // search_count++;
    //                         return node->items[pos].comp.data.value;
    //                     } else if(node->items[pos].is_data == 2){
    //                         auto alexnode = node->items[pos].comp.leaf_node;

    //             // auto end_time_alex_node = std::chrono::high_resolution_clock::now();
    //             // std::chrono::duration<double> alex_node_duration = end_time_alex_node - start_time_alex_node;
    //             // alex_node_time_sum += alex_node_duration.count();  // 累计 alex_node 查找时间

    //             // auto start_time_find_key = std::chrono::high_resolution_clock::now();

    //                         int idx = alexnode->find_key(key);

    //             // auto end_time_find_key = std::chrono::high_resolution_clock::now();
    //             // std::chrono::duration<double> find_key_duration = end_time_find_key - start_time_find_key;
    //             // find_key_time_sum += find_key_duration.count();  // 累计在 alexdatanode 查找 key 的时间
    //             // search_count++;  // 增加搜索次数

    //                         if (idx < 0) {
    //                             exist = false;
    //                             return static_cast<P>(0);
    //                         } else {
    //                             return alexnode->get_payload(idx);
    //                         }
    //                     } else {
    //                         std::cout << "  node->items[pos].is_data  " << node->items[pos].is_data << std::endl;
    //                     }
    //                 }
    //             // }
    //         }
    //     }
    // }
    bool exists(const T& key) const {
        Node* node = root;
        while (true) {
            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                return false;
            } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
                return node->items[pos].comp.data.key == key;
            } else {
                node = node->items[pos].comp.child;
            }
        }
    }
    void bulk_load(const V* vs, int num_keys) {
        if (num_keys == 0) {
            destroy_tree(root);
            root = build_tree_none();
            return;
        }
        if (num_keys == 1) {
            destroy_tree(root);
            root = build_tree_none();
            insert(vs[0]);
            return;
        }
        if (num_keys == 2) {
            destroy_tree(root);
            root = build_tree_two(vs[0].first, vs[0].second, vs[1].first, vs[1].second);
            return;
        }

        RT_ASSERT(num_keys > 2);
        for (int i = 1; i < num_keys; i ++) {
            RT_ASSERT(vs[i].first > vs[i-1].first);
        }

        T* keys = new T[num_keys];
        P* values = new P[num_keys];
        for (int i = 0; i < num_keys; i ++) {
            keys[i] = vs[i].first;
            values[i] = vs[i].second;
        }
        destroy_tree(root);
        std::cout << "before build  " << root << std::endl;
        // root = build_tree_bulk(keys, values, num_keys);
        // root = build_tree_bottom_up(keys, values, num_keys);
        root = build_tree_bottom_up(keys, values, num_keys);
        std::cout << "after build  " << root << std::endl;

        delete[] keys;
        delete[] values;
    }

    bool remove(const T &key) {
        constexpr int MAX_DEPTH = 128;
        Node *path[MAX_DEPTH];
        int path_size = 0;
        Node *parent = nullptr;

        for (Node* node = root; ; ) {
            RT_ASSERT(path_size < MAX_DEPTH);
            path[path_size++] = node;
            // node->size--;
            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->child_bitmap, pos) == 1) {
                parent = node;
                node = node->items[pos].comp.child;
            } else if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                return false;
            } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
                BITMAP_SET(node->none_bitmap, pos);
                for(int i = 0; i < path_size; i++) {
                    path[i]->size--;
                }
                if(node->size == 0) {
                    int parent_pos = PREDICT_POS(parent, key);
                    BITMAP_CLEAR(parent->child_bitmap, parent_pos);
                    BITMAP_SET(parent->none_bitmap, parent_pos);
                    delete_items(node->items, node->num_items);
                    const int bitmap_size = BITMAP_SIZE(node->num_items);
                    delete_bitmap(node->none_bitmap, bitmap_size);
                    delete_bitmap(node->child_bitmap, bitmap_size);
                    delete_nodes(node, 1);
                }
                return true;
            }
        }
    }

    bool update(const T &key, const P& value) {
        for (Node* node = root; ; ) {
            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                return false;
            } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
                //11111111111111111111111111111111111111
                // node->items[pos].comp.data.value = value;
                return true;
            } else {
                node = node->items[pos].comp.child;
            }
        }
    }

    // Find the minimum `len` keys which are no less than `lower`, returns the number of found keys.
    int range_query_len(std::pair<T,P>* results, const T& lower, int len) {
        // return range_core_len<false>(results, 0, root, lower, len);
        return 1;
    }

    void show() const {
        printf("============= SHOW LIPP ================\n");

        std::stack<Node*> s;
        s.push(root);
        while (!s.empty()) {
            Node* node = s.top(); s.pop();

            printf("Node(%p, a = %lf, b = %lf, num_items = %d)", node, node->model.a, node->model.b, node->num_items);
            printf("[");
            int first = 1;
            for (int i = 0; i < node->num_items; i ++) {
                if (!first) {
                    printf(", ");
                }
                first = 0;
                if (BITMAP_GET(node->none_bitmap, i) == 1) {
                    printf("None");
                } else if (BITMAP_GET(node->child_bitmap, i) == 0) {
                    std::stringstream s;
                    s << node->items[i].comp.data.key;
                    printf("Key(%s)", s.str().c_str());
                } else {
                    printf("Child(%p)", node->items[i].comp.child);
                    s.push(node->items[i].comp.child);
                }
            }
            printf("]\n");
        }
    }
    void print_depth() const {
        std::stack<Node*> s;
        std::stack<int> d;
        s.push(root);
        d.push(1);

        int max_depth = 1;
        int sum_depth = 0, sum_nodes = 0;
        while (!s.empty()) {
            Node* node = s.top(); s.pop();
            int depth = d.top(); d.pop();
            for (int i = 0; i < node->num_items; i ++) {
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    s.push(node->items[i].comp.child);
                    d.push(depth + 1);
                } else if (BITMAP_GET(node->none_bitmap, i) != 1) {
                    max_depth = std::max(max_depth, depth);
                    sum_depth += depth;
                    sum_nodes ++;
                }
            }
        }

        printf("max_depth = %d, avg_depth = %.2lf\n", max_depth, double(sum_depth) / double(sum_nodes));
    }
    void verify() const {
        std::stack<Node*> s;
        s.push(root);

        while (!s.empty()) {
            Node* node = s.top(); s.pop();
            int sum_size = 0;
            for (int i = 0; i < node->num_items; i ++) {
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    s.push(node->items[i].comp.child);
                    sum_size += node->items[i].comp.child->size;
                } else if (BITMAP_GET(node->none_bitmap, i) != 1) {
                    sum_size ++;
                }
            }
            RT_ASSERT(sum_size == node->size);
        }
    }
    void print_stats() const {
        printf("======== Stats ===========\n");
        if (USE_FMCD) {
            printf("\t fmcd_success_times = %lld\n", stats.fmcd_success_times);
            printf("\t fmcd_broken_times = %lld\n", stats.fmcd_broken_times);
        }
        #if COLLECT_TIME
        printf("\t time_scan_and_destory_tree = %lf\n", stats.time_scan_and_destory_tree);
        printf("\t time_build_tree_bulk = %lf\n", stats.time_build_tree_bulk);
        #endif
    }
    size_t index_size() const {
        std::stack<Node*> s;
        s.push(root);

        size_t size = 0;
        while (!s.empty()) {
            Node* node = s.top(); s.pop();
            size += sizeof(*node);
            size += sizeof(*(node->none_bitmap));
            size += sizeof(*(node->child_bitmap));
            for (int i = 0; i < node->num_items; i ++) {
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    s.push(node->items[i].comp.child);
                    size += sizeof(Item);
                } else {
                    if (BITMAP_GET(node->none_bitmap, i) == 1) {
                        size += sizeof(Item);
                    } 
                }
            }
        }
        return size;
    }

    //第一层精准预测的 不存到alexdatanode里 第一层冲突的直接存到第二层的alexdatanode里 类似这种的size计算
    size_t total_size() const {
        std::stack < Node * > s;
        s.push(root);

        size_t size = 0;
        size_t leaf_size = 0;
        size_t item_size = 0;
        std::unordered_set<alex::AlexDataNode<T, P>*> calculated_leaf_nodes;  // 记录已经计算过的 leaf_node
        while (!s.empty()) {
            Node *node = s.top();
            // std::cout << "计算size 应该也只输出一次 只有一层  " << node->num_items << std::endl;
            s.pop();
            size += sizeof(*node);
            size += sizeof(*(node->none_bitmap));
            size += sizeof(*(node->child_bitmap));
            for (int i = 0; i < node->num_items; i++) {
                size += sizeof(Item);
                item_size += sizeof(Item);
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    s.push(node->items[i].comp.child);
                } else if(BITMAP_GET(node->none_bitmap, i) == 0) {
                    auto alex_leaf_node = node->items[i].comp.leaf_node;
                    // size += alex_leaf_node->data_size();
                    // leaf_size += alex_leaf_node->data_size();
                    // 检查该 leaf_node 是否已经计算过
                    if (calculated_leaf_nodes.find(alex_leaf_node) == calculated_leaf_nodes.end()) {
                        size += alex_leaf_node->data_size();
                        leaf_size += alex_leaf_node->data_size();
                        calculated_leaf_nodes.insert(alex_leaf_node);  // 记录该 leaf_node 为已计算
                    }
                }
            }
        }
        std::cout << "alexdatanoded的总size leaf_size: " << leaf_size << std::endl;
        std::cout << "item_size: " << item_size << std::endl;
        return size;
    }

    // size_t total_size() const {
    //     std::stack < Node * > s;
    //     s.push(root);

    //     size_t size = 0;
    //     size_t leaf_size = 0;
    //     size_t item_size = 0;
    //     while (!s.empty()) {
    //         Node *node = s.top();
    //         // std::cout << "计算size 应该也只输出一次 只有一层  " << node->num_items << std::endl;
    //         s.pop();
    //         size += sizeof(*node);
    //         size += sizeof(*(node->none_bitmap));
    //         size += sizeof(*(node->child_bitmap));
    //         for (int i = 0; i < node->num_items; i++) {
    //             size += sizeof(Item);
    //             item_size += sizeof(Item);
    //             if (BITMAP_GET(node->child_bitmap, i) == 1) {
    //                 s.push(node->items[i].comp.child);
    //             } else if(BITMAP_GET(node->none_bitmap, i) == 0) {
    //                 auto alex_leaf_node = node->items[i].comp.leaf_node;
    //                 size += alex_leaf_node->data_size();
    //                 leaf_size += alex_leaf_node->data_size();
    //             }
    //         }
    //     }
    //     std::cout << "alexdatanoded的总size leaf_size: " << leaf_size << std::endl;
    //     std::cout << "item_size: " << item_size << std::endl;
    //     return size;
    // }

private:
    struct Node;
    // struct Item
    // {
    //     union {
    //         struct {
    //             T key;
    //             P value;
    //         } data;
    //         Node* child;
    //         alex::AlexDataNode<T,P>* leaf_node;
    //     } comp;
    //     // int is_data = 0;//等于1的时候是直接存的键值对，等于2的时候是存到alexdatanode里
    //     // alex::LinearModel<T> item_model;
    // };
    struct Item
    {
        union {
            //感觉data结构体不需要存在，最后可以删掉，可以减小size 也对cache更友好 一次能抓上来的Item就更多了
            // struct {
            //     T key;
            //     P value;
            // } data;
            Node* child;
            // struct {
            //     std::vector<T> firstkeys;  // 存储 firstkey 的 vector
            //     std::vector<alex::AlexDataNode<T,P>*> datanodes; // 存储指针的 vector
            // } vec_child;
            alex::AlexDataNode<T,P>* leaf_node;
        } comp;
    };
    struct Node
    {
        int is_two; // is special node for only two keys
        int build_size; // tree size (include sub nodes) when node created
        int size; // current tree size (include sub nodes)
        int fixed; // fixed node will not trigger rebuild
        int num_inserts, num_insert_to_data;
        int num_items; // size of items
        LinearModel<T> model;
        Item* items;
        bitmap_t* none_bitmap; // 1 means None, 0 means Data or Child
        bitmap_t* child_bitmap; // 1 means Child. will always be 0 when none_bitmap is 1 没有直接是data的了，所以child_bitmap是0的 就代表指向leafnode
        
    };

    // Item() {
    //     comp.child = nullptr;
    //     comp.leaf_node = nullptr;
    //     comp.vec_child.firstkeys.clear();  // 明确初始化为空
    //     comp.vec_child.datanodes.clear();  // 明确初始化为空
    // };

    // Node* root = new_nodes(1); //原先这里没有等于号后面那些
    Node* root;
    std::stack<Node*> pending_two;
    
    using Compare = alex::AlexCompare;
    using Alloc = std::allocator<std::pair<T, P>>;
    Compare key_less_ = Compare();
    Alloc allocator_ = Alloc();

    typename alex::AlexDataNode<T,P>::alloc_type data_node_allocator() {
        return typename alex::AlexDataNode<T,P>::alloc_type(allocator_);
    }

    void delete_alexdatanode(alex::AlexDataNode<T,P>* node) {
        if (node == nullptr) {
        return;
        } else if (node->is_leaf_) {
        data_node_allocator().destroy(static_cast<alex::AlexDataNode<T,P>*>(node));
        data_node_allocator().deallocate(static_cast<alex::AlexDataNode<T,P>*>(node), 1);
        } 
    }

    std::allocator<Node> node_allocator;
    Node* new_nodes(int n)
    {
        Node* p = node_allocator.allocate(n);
        // std::cout << "Allocated node at: " << p << std::endl;
        RT_ASSERT(p != NULL && p != (Node*)(-1));
        return p;
    }
    void delete_nodes(Node* p, int n)
    {
        node_allocator.deallocate(p, n);
    }

    std::allocator<Item> item_allocator;
    Item* new_items(int n)
    {
        Item* p = item_allocator.allocate(n);
        RT_ASSERT(p != NULL && p != (Item*)(-1));
        return p;
    }
    void delete_items(Item* p, int n)
    {
        item_allocator.deallocate(p, n);
    }

    std::allocator<bitmap_t> bitmap_allocator;
    bitmap_t* new_bitmap(int n)
    {
        bitmap_t* p = bitmap_allocator.allocate(n);
        RT_ASSERT(p != NULL && p != (bitmap_t*)(-1));
        return p;
    }
    void delete_bitmap(bitmap_t* p, int n)
    {
        bitmap_allocator.deallocate(p, n);
    }

    /// build an empty tree
    Node* build_tree_none()
    {
        Node* node = new_nodes(1);
        node->is_two = 0;
        node->build_size = 0;
        node->size = 0;
        node->fixed = 0;
        node->num_inserts = node->num_insert_to_data = 0;
        node->num_items = 1;
        node->model.a = node->model.b = 0;
        node->items = new_items(1);
        // new (node->items) Item();

        node->none_bitmap = new_bitmap(1);
        node->none_bitmap[0] = 0;
        BITMAP_SET(node->none_bitmap, 0);
        node->child_bitmap = new_bitmap(1);
        node->child_bitmap[0] = 0;

        return node;
    }
    /// build a tree with two keys
    Node* build_tree_two(T key1, P value1, T key2, P value2)
    {
        if (key1 > key2) {
            std::swap(key1, key2);
            std::swap(value1, value2);
        }
        RT_ASSERT(key1 < key2);
        static_assert(BITMAP_WIDTH == 8);

        Node* node = NULL;
        if (pending_two.empty()) {
            node = new_nodes(1);
            node->is_two = 1;
            node->build_size = 2;
            node->size = 2;
            node->fixed = 0;
            node->num_inserts = node->num_insert_to_data = 0;

            node->num_items = 8;
            node->items = new_items(node->num_items);
            node->none_bitmap = new_bitmap(1);
            node->child_bitmap = new_bitmap(1);
            node->none_bitmap[0] = 0xff;
            node->child_bitmap[0] = 0;
        } else {
            node = pending_two.top(); pending_two.pop();
        }

        const long double mid1_key = key1;
        const long double mid2_key = key2;

        const double mid1_target = node->num_items / 3;
        const double mid2_target = node->num_items * 2 / 3;

        node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
        node->model.b = mid1_target - node->model.a * mid1_key;
        RT_ASSERT(isfinite(node->model.a));
        RT_ASSERT(isfinite(node->model.b));

        { // insert key1&value1
            int pos = PREDICT_POS(node, key1);
            RT_ASSERT(BITMAP_GET(node->none_bitmap, pos) == 1);
            BITMAP_CLEAR(node->none_bitmap, pos);
            // node->items[pos].comp.data.key = key1;
            // node->items[pos].comp.data.value = value1;
        }
        { // insert key2&value2
            int pos = PREDICT_POS(node, key2);
            RT_ASSERT(BITMAP_GET(node->none_bitmap, pos) == 1);
            BITMAP_CLEAR(node->none_bitmap, pos);
            // node->items[pos].comp.data.key = key2;
            // node->items[pos].comp.data.value = value2;
        }

        return node;
    }
    /// bulk build, _keys must be sorted in asc order.
    Node* build_tree_bulk(T* _keys, P* _values, int _size)
    {
        if (USE_FMCD) {   
            return build_tree_bulk_fmcd(_keys, _values, _size);
        } else {
            return build_tree_bulk_fast(_keys, _values, _size);
        }
    }
    /// bulk build, _keys must be sorted in asc order.
    /// split keys into three parts at each node.
    Node* build_tree_bulk_fast(T* _keys, P* _values, int _size)
    {
        RT_ASSERT(_size > 1);

        typedef struct {
            int begin;
            int end;
            int level; // top level = 1
            Node* node;
        } Segment;
        std::stack<Segment> s;

        Node* ret = new_nodes(1);
        s.push((Segment){0, _size, 1, ret});

        while (!s.empty()) {
            const int begin = s.top().begin;
            const int end = s.top().end;
            const int level = s.top().level;
            Node* node = s.top().node;
            s.pop();

            RT_ASSERT(end - begin >= 2);
            if (end - begin == 2) {
                Node* _ = build_tree_two(_keys[begin], _values[begin], _keys[begin+1], _values[begin+1]);
                memcpy(node, _, sizeof(Node));
                delete_nodes(_, 1);
            } else {
                T* keys = _keys + begin;
                P* values = _values + begin;
                const int size = end - begin;
                const int BUILD_GAP_CNT = compute_gap_count(size);

                node->is_two = 0;
                node->build_size = size;
                node->size = size;
                node->fixed = 0;
                node->num_inserts = node->num_insert_to_data = 0;

                int mid1_pos = (size - 1) / 3;
                int mid2_pos = (size - 1) * 2 / 3;

                RT_ASSERT(0 <= mid1_pos);
                RT_ASSERT(mid1_pos < mid2_pos);
                RT_ASSERT(mid2_pos < size - 1);

                const long double mid1_key =
                        (static_cast<long double>(keys[mid1_pos]) + static_cast<long double>(keys[mid1_pos + 1])) / 2;
                const long double mid2_key =
                        (static_cast<long double>(keys[mid2_pos]) + static_cast<long double>(keys[mid2_pos + 1])) / 2;

                node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
                const double mid1_target = mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;
                const double mid2_target = mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;

                node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
                node->model.b = mid1_target - node->model.a * mid1_key;
                RT_ASSERT(isfinite(node->model.a));
                RT_ASSERT(isfinite(node->model.b));

                const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
                node->model.b += lr_remains;
                node->num_items += lr_remains * 2;

                if (size > 1e6) {
                    node->fixed = 1;
                }

                node->items = new_items(node->num_items);
                const int bitmap_size = BITMAP_SIZE(node->num_items);
                node->none_bitmap = new_bitmap(bitmap_size);
                node->child_bitmap = new_bitmap(bitmap_size);
                memset(node->none_bitmap, 0xff, sizeof(bitmap_t) * bitmap_size);
                memset(node->child_bitmap, 0, sizeof(bitmap_t) * bitmap_size);

                for (int item_i = PREDICT_POS(node, keys[0]), offset = 0; offset < size; ) {
                    int next = offset + 1, next_i = -1;
                    while (next < size) {
                        next_i = PREDICT_POS(node, keys[next]);
                        if (next_i == item_i) {
                            next ++;
                        } else {
                            break;
                        }
                    }
                    if (next == offset + 1) {
                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        //1111111111111111111111111111111111111111111111
                        // node->items[item_i].comp.data.key = keys[offset];
                        // node->items[item_i].comp.data.value = values[offset];
                    } else {
                        // ASSERT(next - offset <= (size+2) / 3);
                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        BITMAP_SET(node->child_bitmap, item_i);
                        node->items[item_i].comp.child = new_nodes(1);
                        s.push((Segment){begin + offset, begin + next, level + 1, node->items[item_i].comp.child});
                    }
                    if (next >= size) {
                        break;
                    } else {
                        item_i = next_i;
                        offset = next;
                    }
                }
            }
        }

        return ret;
    }

    double compute_cost(Node* nodenode, T* _keys, P* _values, int begin2, int end2)
    {
        double cost = 0.0;
        // RT_ASSERT(_size > 1);

        // typedef struct {
        //     int begin;
        //     int end;
        //     int level; // top level = 1
        //     Node* node;
        // } Segment;
        // std::stack<Segment> s;

        // Node* ret = new_nodes(1);
        // s.push((Segment){0, _size, 1, ret});

        // // int total_level = 0;  // 记录所有层级的总和
        // // int node_count = 0;   // 记录节点的总数
        // // int direct_store_count = 0; // 记录第一层直接存储的 Key 数
        // int confict_num = 0;
        // int leaf_node_keys_num = 0;

        // while (!s.empty()) {
            // std::cout << "应该只输出一次 " << std::endl;
            const int begin = begin2;
            const int end = end2;
            const int level = 0;
            Node* node = nodenode;
            // s.pop();


            // RT_ASSERT(end - begin >= 2);
            // if (end - begin == 2) {
            //     std::cout << "不应该进到这里" << std::endl;
            //     Node* _ = build_tree_two(_keys[begin], _values[begin], _keys[begin+1], _values[begin+1]);
            //     memcpy(node, _, sizeof(Node));
            //     delete_nodes(_, 1);
            // } else {
                T* keys = _keys + begin;
                P* values = _values + begin;
                const int size = end - begin;
                const int BUILD_GAP_CNT = compute_gap_count(size);
                // const int BUILD_GAP_CNT = 0;
                // std::cout << "1111111111size: " << size << std::endl;
                // std::cout << "size * static_cast<int>(BUILD_GAP_CNT + 1): " << size * static_cast<int>(BUILD_GAP_CNT + 1) << std::endl;
                const int max_items = 20000000; // 尝试将num_items控制的最大阈值为 20000000

                node->is_two = 0;
                node->build_size = size;
                node->size = size;
                node->fixed = 0;
                node->num_inserts = node->num_insert_to_data = 0;

                // FMCD method
                // Here the implementation is a little different with Algorithm 1 in our paper.
                // In Algorithm 1, U_T should be (keys[size-1-D] - keys[D]) / (L - 2).
                // But according to the derivation described in our paper, M.A should be less than 1 / U_T.
                // So we added a small number (1e-6) to U_T.
                // In fact, it has only a negligible impact of the performance.
                {
                    const int L = size * static_cast<int>(BUILD_GAP_CNT + 1);
                    int i = 0;
                    int D = 1;
                    RT_ASSERT(D <= size-1-D);
                    double Ut = (static_cast<long double>(keys[size - 1 - D]) - static_cast<long double>(keys[D])) /
                                (static_cast<double>(L - 2)) + 1e-6;
                    while (i < size - 1 - D) {
                        while (i + D < size && keys[i + D] - keys[i] >= Ut) {
                            i ++;
                        }
                        if (i + D >= size) {
                            break;
                        }
                        D = D + 1;
                        if (D * 3 > size) break;
                        RT_ASSERT(D <= size-1-D);
                        Ut = (static_cast<long double>(keys[size - 1 - D]) - static_cast<long double>(keys[D])) /
                             (static_cast<double>(L - 2)) + 1e-6;
                    }
                    if (D * 3 <= size) {
                        stats.fmcd_success_times ++;

                        node->model.a = 1.0 / Ut;
                        node->model.b = (L - node->model.a * (static_cast<long double>(keys[size - 1 - D]) +
                                                              static_cast<long double>(keys[D]))) / 2;
                        RT_ASSERT(isfinite(node->model.a));
                        RT_ASSERT(isfinite(node->model.b));
                        node->num_items = L;
                    } else {
                        stats.fmcd_broken_times ++;

                        int mid1_pos = (size - 1) / 3;
                        int mid2_pos = (size - 1) * 2 / 3;

                        RT_ASSERT(0 <= mid1_pos);
                        RT_ASSERT(mid1_pos < mid2_pos);
                        RT_ASSERT(mid2_pos < size - 1);

                        const long double mid1_key = (static_cast<long double>(keys[mid1_pos]) +
                                                      static_cast<long double>(keys[mid1_pos + 1])) / 2;
                        const long double mid2_key = (static_cast<long double>(keys[mid2_pos]) +
                                                      static_cast<long double>(keys[mid2_pos + 1])) / 2;

                        node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
                        const double mid1_target = mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;
                        const double mid2_target = mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;

                        node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
                        node->model.b = mid1_target - node->model.a * mid1_key;
                        RT_ASSERT(isfinite(node->model.a));
                        RT_ASSERT(isfinite(node->model.b));
                    }
                }
                RT_ASSERT(node->model.a >= 0);
                const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
                // std::cout << "lr_remains: " << lr_remains << std::endl;
                node->model.b += lr_remains;
                node->num_items += lr_remains * 2;

                // 调整 node->num_items 以适应阈值，并对模型进行比例调整
                if (node->num_items > max_items) {
                    double scale_factor = static_cast<double>(max_items) / node->num_items;
                    
                    // 对模型的斜率和截距进行比例调整
                    node->model.a *= scale_factor;
                    node->model.b *= scale_factor;

                    // 将 num_items 设为最大阈值
                    node->num_items = max_items;
                }

                // std::cout << "调整后的num_items: " << node->num_items << std::endl;

                if (size > 1e6) {
                    node->fixed = 1;
                }

                node->items = new_items(node->num_items);
                const int bitmap_size = BITMAP_SIZE(node->num_items);
                node->none_bitmap = new_bitmap(bitmap_size);
                node->child_bitmap = new_bitmap(bitmap_size);
                memset(node->none_bitmap, 0xff, sizeof(bitmap_t) * bitmap_size);
                memset(node->child_bitmap, 0, sizeof(bitmap_t) * bitmap_size);

                for (int item_i = PREDICT_POS(node, keys[0]), offset = 0; offset < size; ) {
                    int next = offset + 1, next_i = -1;
                    while (next < size) {
                        next_i = PREDICT_POS(node, keys[next]);
                        if (next_i == item_i) {
                            next ++;
                        } else {
                            break;
                        }
                    }
                    if (next == offset + 1) {
                        // BITMAP_CLEAR(node->none_bitmap, item_i);
                        // node->items[item_i].comp.data.key = keys[offset];
                        // node->items[item_i].comp.data.value = values[offset];
                        // // std::cout << "只有一个节点 精确预测: " << keys[offset] << std::endl;

                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        // BITMAP_SET(node->child_bitmap, item_i);
                        node->items[item_i].comp.child = new_nodes(1);
                        V* value_pairs = new V[1];
                        value_pairs[0] = std::make_pair(keys[offset], values[offset]);
                        auto data_node = new (data_node_allocator().allocate(1))
                            alex::AlexDataNode<T, P>(level + 1, derived_params_.max_data_node_slots,
                                            key_less_, allocator_);
                        // data_node->bulk_load(value_pairs, 1);
                        // node->items[item_i].comp.leaf_node = data_node;
                        // leaf_node_keys_num += 1;        
                        alex::LinearModel<T> data_node_model;
                        alex::AlexDataNode<T,P>::build_model(value_pairs, 1, &data_node_model, params_.approximate_model_computation);
                        alex::DataNodeStats stats;
                        data_node->cost_ = alex::AlexDataNode<T,P>::compute_expected_cost(
                            value_pairs, 1, alex::AlexDataNode<T,P>::kInitDensity_,
                            params_.expected_insert_frac, &data_node_model,
                            params_.approximate_cost_computation, &stats);

                        cost += data_node->cost_; 
                    } else {
                        // confict_num ++;
                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        // BITMAP_SET(node->child_bitmap, item_i);
                        // node->items[item_i].comp.child = new_nodes(1);

                        // if(level == 0){
                        //     BITMAP_SET(node->child_bitmap, item_i);
                        //     node->items[item_i].comp.child = new_nodes(1);
                        //     s.push((Segment){begin + offset, begin + next, level + 1, node->items[item_i].comp.child});
                        // } else {
                            const int num_keys = next - offset;
                            V* value_pairs = new V[num_keys];

                            // 填充数组，将 keys 和 values 转换为 std::pair<T, P>
                            for (int i = 0; i < num_keys; ++i) {
                                value_pairs[i] = std::make_pair(_keys[begin + offset + i], _values[begin + offset + i]);
                            }

                            auto data_node = new (data_node_allocator().allocate(1))
                                alex::AlexDataNode<T, P>(level + 1, derived_params_.max_data_node_slots,
                                                key_less_, allocator_);
                            // data_node->bulk_load(value_pairs, num_keys);
                            // node->items[item_i].comp.leaf_node = data_node;
                            // leaf_node_keys_num += num_keys;

                            alex::LinearModel<T> data_node_model;
                            alex::AlexDataNode<T,P>::build_model(value_pairs, num_keys, &data_node_model, params_.approximate_model_computation);
                            alex::DataNodeStats stats;
                            data_node->cost_ = alex::AlexDataNode<T,P>::compute_expected_cost(
                                value_pairs, num_keys, alex::AlexDataNode<T,P>::kInitDensity_,
                                params_.expected_insert_frac, &data_node_model,
                                params_.approximate_cost_computation, &stats);

                            cost += data_node->cost_;
                        // }


                        // s.push((Segment){begin + offset, begin + next, level + 1, node->items[item_i].comp.child});

                        // alex::LinearModel<T> data_node_model;
                        // alex::AlexDataNode<T,P>::build_model(value_pairs, num_keys, &data_node_model, params_.approximate_model_computation);
                        // alex::DataNodeStats stats;
                        // data_node->cost_ = alex::AlexDataNode<T,P>::compute_expected_cost(
                        //     value_pairs, num_keys, alex::AlexDataNode<T,P>::kInitDensity_,
                        //     params_.expected_insert_frac, &data_node_model,
                        //     params_.approximate_cost_computation, &stats);

                        // std::cout << "看一下datanode_cost的数量级: " << data_node->cost_ << std::endl;  //基本都在个位数
                        
                    }
                    if (next >= size) {
                        break;
                    } else {
                        item_i = next_i;
                        offset = next;
                    }
                }
            // }
        // }
        // std::cout << "多加一层lipp之后 alexdatanode里的cost 不算多的那层lipp的cost和traversetoleaf的cost: " << cost << std::endl;
        return cost;
    }

    void scan_tree(Node* _root, int max_level)
    {
        std::stack < Node * > s;
        s.push(_root);
        int root_data_nodes = 0;           // 记录 _root 节点中符合条件的 item 数量
        int non_root_data_nodes = 0;       // 记录非 _root 节点中符合条件的 item 数量
        bool is_root = true;               // 用于区分是否在根节点

        while (!s.empty()) {
            Node *node = s.top();
            s.pop();
            for (int i = 0; i < node->num_items; i++) {
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    s.push(node->items[i].comp.child);
                } else if(BITMAP_GET(node->none_bitmap, i) == 0) {
                    if (is_root) {
                        root_data_nodes++;
                    } else {
                        non_root_data_nodes++;
                    }
                }
            }
            is_root = false;  // 在检查完根节点后，将标记设为 false
        }
        std::cout << "Root里指向alexdatanode的数量: " << root_data_nodes << std::endl;
        std::cout << "Non-root里指向alexdatanode的数量: " << non_root_data_nodes << std::endl;
    }

    /// bulk build, _keys must be sorted in asc order.
    /// FMCD method.
    //原本的lipp的构造方法
    Node* build_tree_bulk_fmcd(T* _keys, P* _values, int _size)
    {
        RT_ASSERT(_size > 1);

        typedef struct {
            int begin;
            int end;
            int level; // top level = 1
            Node* node;
        } Segment;
        std::stack<Segment> s;

        Node* ret = new_nodes(1);
        // Node* retttt = new_nodes(1);
        // std::cout << "ret " << ret << " retttt " << retttt << std::endl;
        s.push((Segment){0, _size, 1, ret});

        int level_num = 0;
        int max_level = 0;
        // int leaf_at_max_level_count = 0;  // 记录叶子节点在最大层的数量
        // int leaf_level_count = 0; 
        int total_nodes = 0;        // 记录所有节点数量
        int max_level_nodes = 0;    // 记录层数为 max_level 的节点数量

        while (!s.empty()) {
            level_num++;
            const int begin = s.top().begin;
            const int end = s.top().end;
            const int level = s.top().level;
            Node* node = s.top().node;
            s.pop();

            if (level > max_level) {  // 更新最大层数
                max_level = level;
                max_level_nodes = 0;  // 如果 max_level 更新，则重置 max_level_nodes
            }

            // 统计节点
            total_nodes++;
            if (level == max_level) {
                max_level_nodes++;
            }

            RT_ASSERT(end - begin >= 2);
            if (end - begin == 2) {
                // leaf_level_count++;
                Node* _ = build_tree_two(_keys[begin], _values[begin], _keys[begin+1], _values[begin+1]);
                memcpy(node, _, sizeof(Node));
                delete_nodes(_, 1);
            } else {
                T* keys = _keys + begin;
                P* values = _values + begin;
                const int size = end - begin;
                const int BUILD_GAP_CNT = compute_gap_count(size);
                // const int BUILD_GAP_CNT = 2;

                node->is_two = 0;
                node->build_size = size;
                node->size = size;
                node->fixed = 0;
                node->num_inserts = node->num_insert_to_data = 0;

                // FMCD method
                // Here the implementation is a little different with Algorithm 1 in our paper.
                // In Algorithm 1, U_T should be (keys[size-1-D] - keys[D]) / (L - 2).
                // But according to the derivation described in our paper, M.A should be less than 1 / U_T.
                // So we added a small number (1e-6) to U_T.
                // In fact, it has only a negligible impact of the performance.
                {
                    const int L = size * static_cast<int>(BUILD_GAP_CNT + 1);
                    int i = 0;
                    int D = 1;
                    RT_ASSERT(D <= size-1-D);
                    double Ut = (static_cast<long double>(keys[size - 1 - D]) - static_cast<long double>(keys[D])) /
                                (static_cast<double>(L - 2)) + 1e-6;
                    while (i < size - 1 - D) {
                        while (i + D < size && keys[i + D] - keys[i] >= Ut) {
                            i ++;
                        }
                        if (i + D >= size) {
                            break;
                        }
                        D = D + 1;
                        if (D * 3 > size) break;
                        RT_ASSERT(D <= size-1-D);
                        Ut = (static_cast<long double>(keys[size - 1 - D]) - static_cast<long double>(keys[D])) /
                             (static_cast<double>(L - 2)) + 1e-6;
                    }
                    if (D * 3 <= size) {
                        stats.fmcd_success_times ++;

                        node->model.a = 1.0 / Ut;
                        node->model.b = (L - node->model.a * (static_cast<long double>(keys[size - 1 - D]) +
                                                              static_cast<long double>(keys[D]))) / 2;
                        RT_ASSERT(isfinite(node->model.a));
                        RT_ASSERT(isfinite(node->model.b));
                        node->num_items = L;
                    } else {
                        stats.fmcd_broken_times ++;

                        int mid1_pos = (size - 1) / 3;
                        int mid2_pos = (size - 1) * 2 / 3;

                        RT_ASSERT(0 <= mid1_pos);
                        RT_ASSERT(mid1_pos < mid2_pos);
                        RT_ASSERT(mid2_pos < size - 1);

                        const long double mid1_key = (static_cast<long double>(keys[mid1_pos]) +
                                                      static_cast<long double>(keys[mid1_pos + 1])) / 2;
                        const long double mid2_key = (static_cast<long double>(keys[mid2_pos]) +
                                                      static_cast<long double>(keys[mid2_pos + 1])) / 2;

                        node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
                        const double mid1_target = mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;
                        const double mid2_target = mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;

                        node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
                        node->model.b = mid1_target - node->model.a * mid1_key;
                        RT_ASSERT(isfinite(node->model.a));
                        RT_ASSERT(isfinite(node->model.b));
                    }
                }
                RT_ASSERT(node->model.a >= 0);
                const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
                node->model.b += lr_remains;
                node->num_items += lr_remains * 2;

                if (size > 1e6) {
                    node->fixed = 1;
                }

                node->items = new_items(node->num_items);
                const int bitmap_size = BITMAP_SIZE(node->num_items);
                node->none_bitmap = new_bitmap(bitmap_size);
                node->child_bitmap = new_bitmap(bitmap_size);
                memset(node->none_bitmap, 0xff, sizeof(bitmap_t) * bitmap_size);
                memset(node->child_bitmap, 0, sizeof(bitmap_t) * bitmap_size);

                for (int item_i = PREDICT_POS(node, keys[0]), offset = 0; offset < size; ) {
                    int next = offset + 1, next_i = -1;
                    while (next < size) {
                        next_i = PREDICT_POS(node, keys[next]);
                        if (next_i == item_i) {
                            next ++;
                        } else {
                            break;
                        }
                    }
                    if (next == offset + 1) {
                        // leaf_level_count++;
                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        // node->items[item_i].comp.data.key = keys[offset];
                        // node->items[item_i].comp.data.value = values[offset];
                    } else {
                        // ASSERT(next - offset <= (size+2) / 3);
                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        BITMAP_SET(node->child_bitmap, item_i);
                        node->items[item_i].comp.child = new_nodes(1);
                        s.push((Segment){begin + offset, begin + next, level + 1, node->items[item_i].comp.child});
                    }
                    if (next >= size) {
                        break;
                    } else {
                        item_i = next_i;
                        offset = next;
                    }
                }
            }
        }
        std::cout << "上层lipp的层数为: " << max_level << std::endl;
        std::cout << "Total nodes: " << total_nodes << std::endl;
        std::cout << "Nodes at max level (" << max_level << "): " << max_level_nodes << std::endl;
        // scan_tree(ret,max_level);
        return ret;
    }

    Node* insert_build_fmcd(T* _keys, P* _values, int _size)
    {        
        RT_ASSERT(_size > 1);

        // typedef struct {
        //     int begin;
        //     int end;
        //     int level; // top level = 1
        //     Node* node;
        // } Segment;
        // std::stack<Segment> s;

        Node* ret = new_nodes(1);
        Node* retret = new_nodes(1);
        // std::cout << "ret: " << ret << "  retret: " << retret << std::endl;
        // s.push((Segment){0, _size, 1, retret});

        // int level_num = 0;
        // int max_level = 0;
        // // int leaf_at_max_level_count = 0;  // 记录叶子节点在最大层的数量
        // // int leaf_level_count = 0; 
        // int total_nodes = 0;        // 记录所有节点数量
        // int max_level_nodes = 0;    // 记录层数为 max_level 的节点数量

        // while (!s.empty()) {
        //     // level_num++;
            // const int begin = s.top().begin;
            // const int end = s.top().end;
            // const int level = s.top().level;
            // Node* node = s.top().node;
            // // Node* node = new_nodes(1);
            // s.pop();

            const int begin = 0;
            const int end = _size;
            const int level = 1;
            Node* node = retret;

            // if (level > max_level) {  // 更新最大层数
            //     max_level = level;
            //     max_level_nodes = 0;  // 如果 max_level 更新，则重置 max_level_nodes
            // }

            // // 统计节点
            // total_nodes++;
            // if (level == max_level) {
            //     max_level_nodes++;
            // }

            // RT_ASSERT(end - begin >= 2);
            // if (end - begin == 2) {
            //     // // leaf_level_count++;
            //     // Node* _ = build_tree_two(_keys[begin], _values[begin], _keys[begin+1], _values[begin+1]);
            //     // memcpy(node, _, sizeof(Node));
            //     // delete_nodes(_, 1);
            //     std::cout << "不应该存在这种情况" << std::endl;
            // } else {
                T* keys = _keys + begin;
                P* values = _values + begin;
                const int size = end - begin;
                // const int BUILD_GAP_CNT = compute_gap_count(size);
                const int BUILD_GAP_CNT = 2;

                node->is_two = 0;
                node->build_size = size;
                node->size = size;
                node->fixed = 0;
                node->num_inserts = node->num_insert_to_data = 0;

                // FMCD method
                // Here the implementation is a little different with Algorithm 1 in our paper.
                // In Algorithm 1, U_T should be (keys[size-1-D] - keys[D]) / (L - 2).
                // But according to the derivation described in our paper, M.A should be less than 1 / U_T.
                // So we added a small number (1e-6) to U_T.
                // In fact, it has only a negligible impact of the performance.
                {
                    const int L = size * static_cast<int>(BUILD_GAP_CNT + 1);
                    int i = 0;
                    int D = 1;
                    RT_ASSERT(D <= size-1-D);
                    double Ut = (static_cast<long double>(keys[size - 1 - D]) - static_cast<long double>(keys[D])) /
                                (static_cast<double>(L - 2)) + 1e-6;
                    while (i < size - 1 - D) {
                        while (i + D < size && keys[i + D] - keys[i] >= Ut) {
                            i ++;
                        }
                        if (i + D >= size) {
                            break;
                        }
                        D = D + 1;
                        if (D * 3 > size) break;
                        RT_ASSERT(D <= size-1-D);
                        Ut = (static_cast<long double>(keys[size - 1 - D]) - static_cast<long double>(keys[D])) /
                             (static_cast<double>(L - 2)) + 1e-6;
                    }
                    if (D * 3 <= size) {
                        stats.fmcd_success_times ++;

                        node->model.a = 1.0 / Ut;
                        node->model.b = (L - node->model.a * (static_cast<long double>(keys[size - 1 - D]) +
                                                              static_cast<long double>(keys[D]))) / 2;
                        RT_ASSERT(isfinite(node->model.a));
                        RT_ASSERT(isfinite(node->model.b));
                        node->num_items = L;
                    } else {
                        stats.fmcd_broken_times ++;

                        int mid1_pos = (size - 1) / 3;
                        int mid2_pos = (size - 1) * 2 / 3;

                        RT_ASSERT(0 <= mid1_pos);
                        RT_ASSERT(mid1_pos < mid2_pos);
                        RT_ASSERT(mid2_pos < size - 1);

                        const long double mid1_key = (static_cast<long double>(keys[mid1_pos]) +
                                                      static_cast<long double>(keys[mid1_pos + 1])) / 2;
                        const long double mid2_key = (static_cast<long double>(keys[mid2_pos]) +
                                                      static_cast<long double>(keys[mid2_pos + 1])) / 2;

                        node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
                        const double mid1_target = mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;
                        const double mid2_target = mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;

                        node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
                        node->model.b = mid1_target - node->model.a * mid1_key;
                        RT_ASSERT(isfinite(node->model.a));
                        RT_ASSERT(isfinite(node->model.b));
                    }
                }
                RT_ASSERT(node->model.a >= 0);
                const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
                node->model.b += lr_remains;
                node->num_items += lr_remains * 2;

                if (size > 1e6) {
                    node->fixed = 1;
                }

                node->items = new_items(node->num_items);
                const int bitmap_size = BITMAP_SIZE(node->num_items);
                node->none_bitmap = new_bitmap(bitmap_size);
                node->child_bitmap = new_bitmap(bitmap_size);
                memset(node->none_bitmap, 0xff, sizeof(bitmap_t) * bitmap_size);
                memset(node->child_bitmap, 0, sizeof(bitmap_t) * bitmap_size);

                    // if(std::abs(node->model.a - 0.741327) < 1e-6){
                    //     std::cout << "fmcd node->model.a: " << node->model.a << "  node->model.b: " << node->model.b << "  node: " << node << std::endl;
                    //     // if(std::abs(node->model.a - 0.000239909) < 1e-6 && pos == 52493){
                    //         // 遍历 keys 数组以检查是否包含 218852026
                    //         bool contains_key = false;
                    //         int popo = -1;
                    //         for (int i = 0; i < _size; ++i) {
                    //             if (keys[i] == 218852026) {
                    //                 contains_key = true;
                    //                 popo = i;
                    //                 break;  // 找到目标值，提前退出循环
                    //             }
                    //         }

                    //         if (contains_key) {
                    //             // 如果 keys 中包含 218852026，可以在这里执行特定的操作
                    //             std::cout << "fmcd Keys array contains 218852026." << "  i  " << popo << std::endl;
                    //         } else {
                    //             std::cout << "fmcd Keys array does not contain 218852026." << std::endl;
                    //         }
                    //         std::cout << "fmcd Keys array size." << _size << std::endl;
                    //     // }
                    // }

                for (int item_i = PREDICT_POS(node, keys[0]), offset = 0; offset < size; ) {
                    int next = offset + 1, next_i = -1;
                    while (next < size) {
                        next_i = PREDICT_POS(node, keys[next]);
                        if (next_i == item_i) {
                            next ++;
                        } else {
                            break;
                        }
                    }
                    // if(std::abs(node->model.a - 0.741327) < 1e-6){
                    //     // std::cout << "node->model.a: " << node->model.a << "  node->model.b: " << node->model.b << std::endl;
                    //     //上面next++是可能错过输出一些offset值的 
                    //     std::cout << "keys[offset] " << keys[offset] << " offset  " << offset << std::endl;
                    // }
                    // if (next == offset + 1) {
                    //     // leaf_level_count++;
                    //     BITMAP_CLEAR(node->none_bitmap, item_i);
                    //     // node->items[item_i].comp.data.key = keys[offset];
                    //     // node->items[item_i].comp.data.value = values[offset];
                    // } else {
                    //     // ASSERT(next - offset <= (size+2) / 3);
                    //     BITMAP_CLEAR(node->none_bitmap, item_i);
                    //     BITMAP_SET(node->child_bitmap, item_i);
                    //     node->items[item_i].comp.child = new_nodes(1);
                    //     s.push((Segment){begin + offset, begin + next, level + 1, node->items[item_i].comp.child});
                    // }

                    BITMAP_CLEAR(node->none_bitmap, item_i);
                    const int num_keys = next - offset;
                    V* value_pairs = new V[num_keys];

                    // if(std::abs(node->model.a - 0.741327) < 1e-6 && offset == 777){
                    //     std::cout << "next " << next << " num_keys  " << num_keys << std::endl;
                    // }

                    // 填充数组，将 keys 和 values 转换为 std::pair<T, P>
                    for (int i = 0; i < num_keys; ++i) {
                        value_pairs[i] = std::make_pair(_keys[begin + offset + i], _values[begin + offset + i]);
                                    // if(std::abs(node->model.a - 0.741327) < 1e-6){
                                    //     if(item_i == 0){
                                    //         std::cout << "_keys[begin + offset + i] " << _keys[begin + offset + i] << std::endl;
                                    //     }
                                    // }
                    }

                    auto data_node = new (data_node_allocator().allocate(1))
                        alex::AlexDataNode<T, P>(level + 1, derived_params_.max_data_node_slots,
                                        key_less_, allocator_);
                    data_node->bulk_load(value_pairs, num_keys);
                    node->items[item_i].comp.leaf_node = data_node;
                                                        // if(std::abs(node->model.a - 0.741327) < 1e-6){
                                                        //     if(item_i == 0 || item_i == 1 || item_i == 12161){
                                                        //         std::cout << "item_i " << item_i<< std::endl;
                                                        //         std::cout << "111BITMAP_GET(node->none_bitmap, item_i) " << BITMAP_GET(node->none_bitmap, item_i)<< std::endl;
                                                        //         std::cout << "111BITMAP_GET(node->child_bitmap, item_i) " << BITMAP_GET(node->child_bitmap, item_i)<< std::endl;
                                                        //         std::cout << "111node " << node << "  node->items[item_i].comp.leaf_node " << node->items[item_i].comp.leaf_node << "  data_node " << data_node << std::endl;                    
                                                        //     }
                                                        //     // std::cout << "item_i " << item_i<< std::endl;
                                                        // }
                    // if(bug == 1){
                    //     std::cout << "bug node " << node << " item_i " << item_i << " data_node " << data_node << std::endl;
                    // }


                    if (next >= size) {
                        break;
                    } else {
                        item_i = next_i;
                        offset = next;
                    }
                // }
                                                // if(std::abs(node->model.a - 0.741327) < 1e-6){
                                                //     std::cout << "BITMAP_GET(node->none_bitmap, 0) " << BITMAP_GET(node->none_bitmap, 0)<< std::endl;
                                                //     std::cout << "BITMAP_GET(node->child_bitmap, 0) " << BITMAP_GET(node->child_bitmap, 0)<< std::endl;
                                                // }
            }
        // }

        // std::cout << "上层lipp的层数为: " << max_level << std::endl;
        // std::cout << "Total nodes: " << total_nodes << std::endl;
        // std::cout << "Nodes at max level (" << max_level << "): " << max_level_nodes << std::endl;
        // scan_tree(ret,max_level);
        return retret;
        // return node;
    }

    // Node* build_tree_vec_child(T* _keys, P* _values, int _size)
    // {
    //     RT_ASSERT(_size > 1);

    //     typedef struct {
    //         int begin;
    //         int end;
    //         int level; // top level = 1
    //         Node* node;
    //     } Segment;
    //     std::stack<Segment> s;

    //     Node* ret = new_nodes(1);
    //     s.push((Segment){0, _size, 1, ret});

    //     int level_num = 0;
    //     int max_level = 0;
    //     // int leaf_at_max_level_count = 0;  // 记录叶子节点在最大层的数量
    //     // int leaf_level_count = 0; 
    //     int total_nodes = 0;        // 记录所有节点数量
    //     int max_level_nodes = 0;    // 记录层数为 max_level 的节点数量

    //     while (!s.empty()) {
    //         level_num++;
    //         const int begin = s.top().begin;
    //         const int end = s.top().end;
    //         const int level = s.top().level;
    //         Node* node = s.top().node;
    //         s.pop();

    //         if (level > max_level) {  // 更新最大层数
    //             max_level = level;
    //             max_level_nodes = 0;  // 如果 max_level 更新，则重置 max_level_nodes
    //         }

    //         // 统计节点
    //         total_nodes++;
    //         if (level == max_level) {
    //             max_level_nodes++;
    //         }

    //         RT_ASSERT(end - begin >= 2);
    //         if (end - begin == 2) {
    //             // leaf_level_count++;
    //             Node* _ = build_tree_two(_keys[begin], _values[begin], _keys[begin+1], _values[begin+1]);
    //             memcpy(node, _, sizeof(Node));
    //             delete_nodes(_, 1);
    //         } else {
    //             T* keys = _keys + begin;
    //             P* values = _values + begin;
    //             const int size = end - begin;
    //             const int BUILD_GAP_CNT = compute_gap_count(size);

    //             node->is_two = 0;
    //             node->build_size = size;
    //             node->size = size;
    //             node->fixed = 0;
    //             node->num_inserts = node->num_insert_to_data = 0;

    //             // FMCD method
    //             // Here the implementation is a little different with Algorithm 1 in our paper.
    //             // In Algorithm 1, U_T should be (keys[size-1-D] - keys[D]) / (L - 2).
    //             // But according to the derivation described in our paper, M.A should be less than 1 / U_T.
    //             // So we added a small number (1e-6) to U_T.
    //             // In fact, it has only a negligible impact of the performance.
    //             {
    //                 const int L = size * static_cast<int>(BUILD_GAP_CNT + 1);
    //                 int i = 0;
    //                 int D = 1;
    //                 RT_ASSERT(D <= size-1-D);
    //                 double Ut = (static_cast<long double>(keys[size - 1 - D]) - static_cast<long double>(keys[D])) /
    //                             (static_cast<double>(L - 2)) + 1e-6;
    //                 while (i < size - 1 - D) {
    //                     while (i + D < size && keys[i + D] - keys[i] >= Ut) {
    //                         i ++;
    //                     }
    //                     if (i + D >= size) {
    //                         break;
    //                     }
    //                     D = D + 1;
    //                     if (D * 3 > size) break;
    //                     RT_ASSERT(D <= size-1-D);
    //                     Ut = (static_cast<long double>(keys[size - 1 - D]) - static_cast<long double>(keys[D])) /
    //                          (static_cast<double>(L - 2)) + 1e-6;
    //                 }
    //                 if (D * 3 <= size) {
    //                     stats.fmcd_success_times ++;

    //                     node->model.a = 1.0 / Ut;
    //                     node->model.b = (L - node->model.a * (static_cast<long double>(keys[size - 1 - D]) +
    //                                                           static_cast<long double>(keys[D]))) / 2;
    //                     RT_ASSERT(isfinite(node->model.a));
    //                     RT_ASSERT(isfinite(node->model.b));
    //                     node->num_items = L;
    //                 } else {
    //                     stats.fmcd_broken_times ++;

    //                     int mid1_pos = (size - 1) / 3;
    //                     int mid2_pos = (size - 1) * 2 / 3;

    //                     RT_ASSERT(0 <= mid1_pos);
    //                     RT_ASSERT(mid1_pos < mid2_pos);
    //                     RT_ASSERT(mid2_pos < size - 1);

    //                     const long double mid1_key = (static_cast<long double>(keys[mid1_pos]) +
    //                                                   static_cast<long double>(keys[mid1_pos + 1])) / 2;
    //                     const long double mid2_key = (static_cast<long double>(keys[mid2_pos]) +
    //                                                   static_cast<long double>(keys[mid2_pos + 1])) / 2;

    //                     node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
    //                     const double mid1_target = mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;
    //                     const double mid2_target = mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;

    //                     node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
    //                     node->model.b = mid1_target - node->model.a * mid1_key;
    //                     RT_ASSERT(isfinite(node->model.a));
    //                     RT_ASSERT(isfinite(node->model.b));
    //                 }
    //             }
    //             RT_ASSERT(node->model.a >= 0);
    //             const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
    //             node->model.b += lr_remains;
    //             node->num_items += lr_remains * 2;

    //             if (size > 1e6) {
    //                 node->fixed = 1;
    //             }

    //             node->items = new_items(node->num_items);
    //             const int bitmap_size = BITMAP_SIZE(node->num_items);
    //             node->none_bitmap = new_bitmap(bitmap_size);
    //             node->child_bitmap = new_bitmap(bitmap_size);
    //             memset(node->none_bitmap, 0xff, sizeof(bitmap_t) * bitmap_size);
    //             memset(node->child_bitmap, 0, sizeof(bitmap_t) * bitmap_size);

    //             for (int item_i = PREDICT_POS(node, keys[0]), offset = 0; offset < size; ) {
    //                 int next = offset + 1, next_i = -1;
    //                 while (next < size) {
    //                     next_i = PREDICT_POS(node, keys[next]);
    //                     if (next_i == item_i) {
    //                         next ++;
    //                     } else {
    //                         break;
    //                     }
    //                 }
    //                 if (next == offset + 1) {
    //                     // leaf_level_count++;
    //                     BITMAP_CLEAR(node->none_bitmap, item_i);
    //                     node->items[item_i].comp.data.key = keys[offset];
    //                     node->items[item_i].comp.data.value = values[offset];
    //                 } else {
    //                     // ASSERT(next - offset <= (size+2) / 3);
    //                     BITMAP_CLEAR(node->none_bitmap, item_i);
    //                     BITMAP_SET(node->child_bitmap, item_i);
    //                     // node->items[item_i].comp.child = new_nodes(1);
    //                     // s.push((Segment){begin + offset, begin + next, level + 1, node->items[item_i].comp.child});
    //                 }
    //                 if (next >= size) {
    //                     break;
    //                 } else {
    //                     item_i = next_i;
    //                     offset = next;
    //                 }
    //             }
    //         }
    //     }
    //     std::cout << "上层lipp的层数为: " << max_level << std::endl;
    //     std::cout << "Total nodes: " << total_nodes << std::endl;
    //     std::cout << "Nodes at max level (" << max_level << "): " << max_level_nodes << std::endl;
    //     return ret;
    // }

    // //多个指针指向同一个叶子节点，以alex划分的叶子节点为基础，再根据上层重新拆分节点，不合并，使得上层不会将不同叶子节点里的key映射到同一个slot里
    // //一个slot里可以存多个指针指向叶子节点
    // Node* build_tree_bottom_up(T* _keys, P* _values, int _size)
    // {
    //     RT_ASSERT(_size > 1);

    //     std::vector<T> first_keys;
    //     //用PGM的分段方式
    //     std::vector<std::pair<T, P>> key_value;
    //     key_value.reserve(_size);  // 预分配空间以提高性能
    //     for (int i = 0; i < _size; ++i) {
    //         key_value.emplace_back(_keys[i], _values[i]);
    //     }
    //     first_keys = lial::internal::segment_linear_optimal_model_fk(key_value, _size, 4);

    //     std::cout << "first_keys.size: " << first_keys.size() << std::endl;
        
    //     Node * build_root = build_tree_bulk_fmcd(first_keys.data(), _values, first_keys.size());
    //     // Node * build_root = build_tree_vec_child(first_keys.data(), _values, first_keys.size());

    //     // std::cout << "build_tree_bulk_fmcd: " << build_root << std::endl;

    //     int segment_count = first_keys.size();

    //     int modified_count = 0;
    //     std::vector<bool> modified_flags(segment_count + 1, false);
        
    //     for (int i = 2; i < segment_count+1; ++i) {
    //         int start_idx = std::distance(_keys, std::lower_bound(_keys, _keys + _size, first_keys[i - 1]));
    //         // int end_idx = std::distance(_keys, std::lower_bound(_keys, _keys + _size, first_keys[i]));
    
    //         // 如果当前处理的是最后一个区间，则将 end_idx 设置为 _keys 的结尾
    //         int end_idx;
    //         if (i - 1 == segment_count - 1) {
    //             end_idx = _size;  // 设置为 _keys 的最后一个索引 + 1
    //         } else {
    //             end_idx = std::distance(_keys, std::lower_bound(_keys, _keys + _size, first_keys[i]));
    //         }

    //         for (int j = start_idx; j < end_idx; ++j) {
    //             auto [node_prev, item_prev] = build_at(build_root, _keys[start_idx-1]);
    //             auto [node_current, item_current] = build_at(build_root, _keys[j]);

    //             if (node_prev == node_current && item_prev == item_current) {

    //                 if (!modified_flags[i - 1]) {  // 如果该 first_keys[i-1] 尚未被标记为修改
    //                     modified_flags[i - 1] = true;
    //                     modified_count++;  // 记录该 first_keys[i-1] 为被修改的唯一计数
    //                 }

    //                 if(j+1 == end_idx){
    //                     first_keys[i-1] = -1;
    //                     break;
    //                 } else {
    //                     first_keys[i-1] = _keys[j+1];
    //                 }
    //             }
    //         }
    //     }
    //     // std::cout << "到这之前没问题 " << std::endl;
    //     // std::cout << "build_root " << build_root << std::endl;
    //     int current_index = 0;
    //     for (int i = 0; i < segment_count; i++) {
    //         T start_key = first_keys[i];
    //         T end_key;

    //         // 如果start_key为-1，跳过当前循环
    //         if (start_key == -1) {
    //             continue;
    //         }

    //         // 如果当前是最后一个有效的 start_key，则 end_key 设置为比最后一个元素大1
    //         if (i == first_keys.size() - 1) {
    //             end_key = _keys[_size - 1] + 1;
    //         } else {
    //             // 否则，寻找下一个有效的 end_key
    //             end_key = first_keys[i + 1];
    //             int j = i + 1;
    //             while (end_key == -1 && j < first_keys.size() - 1) {
    //                 ++j;
    //                 end_key = first_keys[j];
    //             }

    //             // 如果没有找到有效的end_key，则扩展到_keys的末尾
    //             if (end_key == -1) {
    //                 end_key = _keys[_size - 1] + 1;
    //             }
    //         }

    //         // 筛选出位于 (start_key, end_key) 区间内的 key-value 对
    //         std::vector<std::pair<T, P>> value_pairs;
    //         while (current_index < _size && _keys[current_index] < end_key) {
    //             if (_keys[current_index] >= start_key) {
    //                 value_pairs.emplace_back(_keys[current_index], _values[current_index]);
    //             }
    //             current_index++;
    //         }

    //         int num_keys = value_pairs.size();
    //         auto data_node = new (data_node_allocator().allocate(1))
    //             alex::AlexDataNode<T, P>(1, derived_params_.max_data_node_slots, key_less_, allocator_);
    //         if (num_keys > 0) {                
    //             // 执行批量加载
    //             data_node->bulk_load(value_pairs.data(), num_keys);
    //         }

    //         // 遍历 data_node 里的每个 key 并设置 item_i
    //         int reserve_size = value_pairs.size();
    //         int previous_item_i = -1;
    //         for (const auto& [key, value] : value_pairs) {
    //             // auto [node, item_i] = build_at(build_root, key);
    //             auto [is_vec_child, item_i] = build_simd_at(build_root, key);
    //             BITMAP_CLEAR(build_root->none_bitmap, item_i);
    //             // node->items[item_i].comp.leaf_node = data_node;  // 为每个 key 设置 leaf_node
    //             if (item_i == previous_item_i) {
    //                 continue;
    //             }
    //             previous_item_i = item_i;
    //             if(is_vec_child == 0){
    //                 build_root->items[item_i].comp.leaf_node = data_node;
    //             } else {
    //                 // build_root->items[item_i].comp.vec_child.firstkeys.push_back(start_key);
    //                 // build_root->items[item_i].comp.vec_child.datanodes.push_back(data_node);
    //                 //判断item_i跟之前一不一样就行吧 一样的话 就不用重新push了啊
    //                 // std::cout << "111这里的问题？ " << std::endl;
    //                 // 查找是否已经存在相同的 data_node
    //                 auto& firstkeys = build_root->items[item_i].comp.vec_child.firstkeys;
    //                 auto& datanodes = build_root->items[item_i].comp.vec_child.datanodes;

    //                 // if (firstkeys.capacity() < reserve_size) {
    //                     // firstkeys.reserve(reserve_size);
    //                 //     datanodes.reserve(reserve_size);
    //                 // }
                    
    //                 // bool found = false;
    //                 // for (int i = 0; i < datanodes.size(); ++i) {
    //                 //     if (datanodes[i] == data_node) {
    //                 //         // // 如果 data_node 已经存在，更新 firstkey
    //                 //         // firstkeys[i] = start_key;
    //                 //         found = true;
    //                 //         break;
    //                 //     }
    //                 // }
    //             //     std::cout << "222这里的问题？ " << build_root << std::endl;

    //             // if (build_root == nullptr) {
    //             //     std::cerr << "build_root not properly initialized." << std::endl;
    //             // }
    //             // if (build_root->items[item_i].comp.vec_child.firstkeys.empty()) {
    //             //     std::cerr << "items[item_i] not properly initialized." << std::endl;
    //             // }


    //                 // // 如果 data_node 不存在，追加新的 entry
    //                 // if (!found) {
    //                 // try {
    //                 //     firstkeys.clear();
    //                 //     firstkeys.push_back(start_key);
    //                 //     // firstkeys.push_back(1);
    //                 // } catch (const std::bad_alloc& e) {
    //                 //     std::cerr << "Memory allocation failed during push_back: " << e.what() << std::endl;
    //                 //     std::cerr << "firstkeys.size(): " << firstkeys.size() << std::endl;
    //                 //     throw; // 重新抛出异常，或根据需要处理
    //                 // }
    //                     // firstkeys.clear();
    //                     // datanodes.clear();
    //                     if(firstkeys.size() > 200000000){
    //                         firstkeys.clear();
    //                     }
    //                     if(datanodes.size() > 200000000){
    //                         datanodes.clear();
    //                     }

    //                     firstkeys.push_back(start_key);
    //                     datanodes.push_back(data_node);
    //                 // }
    //                 // std::cout << "333这里的问题？ " << std::endl;
    //             }
    //         }

    //     }
    //     return build_root;
    // }

    //多个指针指向同一个叶子节点，以alex划分的叶子节点为基础，再根据上层重新划分，使得上层不会将不同叶子节点里的key映射到同一个slot里
    //这个是目前吞吐最好的 误差64 gap=2 best_2
    Node* build_tree_bottom_up(T* _keys, P* _values, int _size)
    {
        RT_ASSERT(_size > 1);

        // std::vector<T> first_keys;
        std::vector<std::pair<T, P>> fk_values;
                            //直接用alex的分段
                            // std::ifstream infile("alex_libio_first_keys.txt");    // 打开文件

                            // if (!infile) {
                            //     std::cerr << "Unable to open file: " << "alex_first_keys.txt" << std::endl;
                            // }

                            // T key;
                            // while (infile >> key) {            // 从文件中逐个读取 KEY_TYPE 值
                            //     // std::cout << "Read key: " << key << std::endl;
                            //     first_keys.push_back(key);     // 将读取到的值存入 vector
                            // }

                            // infile.close();                    // 关闭文件
                            // std::cout << "first_keys.size: " << first_keys.size() << std::endl;

        //用PGM的分段方式
        std::vector<std::pair<T, P>> key_value;
        key_value.reserve(_size);  // 预分配空间以提高性能
        for (int i = 0; i < _size; ++i) {
            key_value.emplace_back(_keys[i], _values[i]);
        }
        // first_keys = lial::internal::segment_linear_optimal_model_fk(key_value, _size, 64);
        fk_values = lial::internal::segment_linear_optimal_model_fk_value(key_value, _size, 64);

        std::cout << "first_keys.size: " << fk_values.size() << std::endl;
        int fk_size = fk_values.size();
        std::vector<T> first_keys(fk_size);
        // 提取 fk_values 中的第一个键，并存储到 first_keys
        for (size_t i = 0; i < fk_size; ++i) {
            first_keys[i] = fk_values[i].first; // 提取每个段的第一个键
        }

        Node * build_root;
        if (fk_size == 1) {
            build_root = build_tree_none();
        } else {
            build_root = build_tree_bulk_fmcd(first_keys.data(), _values, first_keys.size());
        }
        
        // Node * build_root = build_tree_bulk_fmcd(first_keys.data(), _values, first_keys.size());

        std::cout << "build_tree_bulk_fmcd: " << build_root << std::endl;
        // std::cout << "build_tree_bottom_up " << "BITMAP_GET(build_root->none_bitmap, 0) " << BITMAP_GET(build_root->none_bitmap, 0) << " BITMAP_GET(build_root->child_bitmap, 0) " << BITMAP_GET(build_root->child_bitmap, 0) << std::endl;

        int segment_count = first_keys.size();

        int modified_count = 0;
        std::vector<bool> modified_flags(segment_count + 1, false);
        
        for (int i = 2; i < segment_count+1; ++i) {
            int start_idx = std::distance(_keys, std::lower_bound(_keys, _keys + _size, first_keys[i - 1]));
            // int end_idx = std::distance(_keys, std::lower_bound(_keys, _keys + _size, first_keys[i]));
    
            // 如果当前处理的是最后一个区间，则将 end_idx 设置为 _keys 的结尾
            int end_idx;
            if (i - 1 == segment_count - 1) {
                end_idx = _size;  // 设置为 _keys 的最后一个索引 + 1
            } else {
                end_idx = std::distance(_keys, std::lower_bound(_keys, _keys + _size, first_keys[i]));
            }

            for (int j = start_idx; j < end_idx; ++j) {
                auto [node_prev, item_prev] = build_at(build_root, _keys[start_idx-1]);
                auto [node_current, item_current] = build_at(build_root, _keys[j]);

                if (node_prev == node_current && item_prev == item_current) {

                    if (!modified_flags[i - 1]) {  // 如果该 first_keys[i-1] 尚未被标记为修改
                        modified_flags[i - 1] = true;
                        modified_count++;  // 记录该 first_keys[i-1] 为被修改的唯一计数
                    }

                    if(j+1 == end_idx){
                        first_keys[i-1] = -1;
                        break;
                    } else {
                        first_keys[i-1] = _keys[j+1];
                    }
                }
            }
        }


        int aaa = 0;
        for (int i = 0; i < segment_count; i++) {
            if(first_keys[i] == -1){
                aaa++;
            }
        }
        std::cout << "重组后被合并了的叶子节点个数：  " << aaa << " 被修改了的firstkey的个数 " << modified_count << std::endl;


        int current_index = 0;
        for (int i = 0; i < segment_count; i++) {
            T start_key = first_keys[i];
            T end_key;

            // 如果start_key为-1，跳过当前循环
            if (start_key == -1) {
                continue;
            }

            // 如果当前是最后一个有效的 start_key，则 end_key 设置为比最后一个元素大1
            if (i == first_keys.size() - 1) {
                end_key = _keys[_size - 1] + 1;
            } else {
                // 否则，寻找下一个有效的 end_key
                end_key = first_keys[i + 1];
                int j = i + 1;
                while (end_key == -1 && j < first_keys.size() - 1) {
                    ++j;
                    end_key = first_keys[j];
                }

                // 如果没有找到有效的end_key，则扩展到_keys的末尾
                if (end_key == -1) {
                    end_key = _keys[_size - 1] + 1;
                }
            }

            // 筛选出位于 (start_key, end_key) 区间内的 key-value 对
            std::vector<std::pair<T, P>> value_pairs;
            while (current_index < _size && _keys[current_index] < end_key) {
                if (_keys[current_index] >= start_key) {
                    value_pairs.emplace_back(_keys[current_index], _values[current_index]);
                }
                current_index++;
            }

            int num_keys = value_pairs.size();
            auto data_node = new (data_node_allocator().allocate(1))
                alex::AlexDataNode<T, P>(1, derived_params_.max_data_node_slots, key_less_, allocator_);
            if (num_keys > 0) {                
                // 执行批量加载
                data_node->bulk_load(value_pairs.data(), num_keys);
            }
            // std::cout << "  data_node: " << data_node << "  current_index: " << current_index << std::endl;
            // 遍历 data_node 里的每个 key 并设置 item_i
            for (const auto& [key, value] : value_pairs) {
                // std::cout << "build_root: " << build_root << "  key: " << key << std::endl;
                auto [node, item_i] = build_at(build_root, key);
                    // std::cout << "  node: " << node << "  item_i: " << item_i << std::endl;
                BITMAP_CLEAR(node->none_bitmap, item_i);
                node->items[item_i].comp.leaf_node = data_node;  // 为每个 key 设置 leaf_node
            }

        }
        return build_root;
    }




    // //is_data的bulk实现方法如下
    // Node* build_tree_bottom_up(T* _keys, P* _values, int _size)
    // {
    //     RT_ASSERT(_size > 1);

    //     std::vector<T> first_keys;
    //     std::ifstream infile("alex_libio_first_keys.txt");    // 打开文件

    //     if (!infile) {
    //         std::cerr << "Unable to open file: " << "alex_first_keys.txt" << std::endl;
    //     }

    //     T key;
    //     while (infile >> key) {            // 从文件中逐个读取 KEY_TYPE 值
    //         // std::cout << "Read key: " << key << std::endl;
    //         first_keys.push_back(key);     // 将读取到的值存入 vector
    //     }

    //     infile.close();                    // 关闭文件
    //     // std::cout << "first_keys.size: " << first_keys.size() << std::endl;

    //     Node * build_root = build_tree_bulk_fmcd(first_keys.data(), _values, first_keys.size());
    //     int leaf_node_keys_num = 0;
    //     int exact_key_num = 0;

    //     int temp = 0;
    //     for (int offset = 0; offset < _size; ) {
    //         auto [node, item_i] = build_at(build_root, _keys[offset]);
    //         int next = offset + 1, next_i = -1;
    //         while (next < _size) {
    //             auto [node2, next_i] = build_at(build_root, _keys[next]);
    //             if (next_i == item_i && node == node2) {
    //                 next ++;
    //             } else {
    //                 break;
    //             }
    //         }
    //         if (next == offset + 1) {
    //             BITMAP_CLEAR(node->none_bitmap, item_i);
    //             node->items[item_i].is_data = 1;
    //             node->items[item_i].comp.data.key = _keys[offset];
    //             node->items[item_i].comp.data.value = _values[offset];
    //                     // if(_keys[offset] == 216652112){
    //                     //     std::cout << "存在这个精准预测: " << _keys[offset] << " node->items[item_i].comp.data.key  " << node->items[item_i].comp.data.key << std::endl;
    //                     // }
    //                     // if(temp <= 2){
    //                     //     std::cout << "输出一个精准预测: " << _keys[offset] << " node->items[item_i].comp.data.key  " << node->items[item_i].comp.data.key << std::endl;
    //                     //     temp ++;
    //                     // }
    //             // V* value_pairs = new V[1];
    //             // value_pairs[0] = std::make_pair(_keys[offset], _values[offset]);
    //             // auto data_node = new (data_node_allocator().allocate(1))
    //             //     alex::AlexDataNode<T, P>(1, derived_params_.max_data_node_slots,
    //             //                     key_less_, allocator_);//level先随便写吧，后面如果重要，可以在build_at里获得
    //             // data_node->bulk_load(value_pairs, 1);
    //             // node->items[item_i].comp.leaf_node = data_node;
    //             leaf_node_keys_num += 1;
    //             exact_key_num += 1;
    //             // node->items[item_i].is_data = 1; 
    //         } else {
    //             BITMAP_CLEAR(node->none_bitmap, item_i);
    //             node->items[item_i].is_data = 2;
    //             const int num_keys = next - offset;
    //             V* value_pairs = new V[num_keys];

    //             // 填充数组，将 keys 和 values 转换为 std::pair<T, P>
    //             for (int i = 0; i < num_keys; ++i) {
    //                 value_pairs[i] = std::make_pair(_keys[offset + i], _values[offset + i]);
    //                 // if(_keys[offset+i] == 216652112){
    //                 //     auto [node222, item_1] = build_at(build_root, 216652112);
    //                 //     std::cout << "这个数存到alexdatanode里了: " << _keys[offset] << " node->items[item_i].is_data " << node->items[item_i].is_data << " node222->items[item_1].is_data " << node222->items[item_1].is_data << std::endl;
    //                 //     std::cout << "node222: " << node222 << " item_1 " << item_1 << std::endl;
    //                 // }
    //             }

    //             auto data_node = new (data_node_allocator().allocate(1))
    //                 alex::AlexDataNode<T, P>(1, derived_params_.max_data_node_slots,
    //                                 key_less_, allocator_);
    //             // confict_num ++;
    //             data_node->bulk_load(value_pairs, num_keys);
    //             node->items[item_i].comp.leaf_node = data_node;
    //             leaf_node_keys_num += num_keys;
    //             // node->items[item_i].is_data = 2;                       
    //         }
    //         if (next >= _size) {
    //             break;
    //         } else {
    //             // item_i = next_i;
    //             offset = next;
    //         }
    //     }
    //     // std::cout << "111leaf_node_keys_num: " << leaf_node_keys_num << std::endl;
    //     // std::cout << "111exact_key_num: " << exact_key_num << std::endl;
    //     // auto [node2223, item_2] = build_at(build_root, 216652112);
    //     // std::cout << "node2223->items[item_2].is_data " << node2223->items[item_2].is_data << std::endl;
    //     // std::cout << "node2223: " << node2223 << " item_2 " << item_2 << std::endl;

    //     // auto [node4, item_4] = build_at(build_root, 37597);
    //     // std::cout << "node4->items[item_4].is_data " << node4->items[item_4].is_data << " node4->items[item_4].comp.data.key " << node4->items[item_4].comp.data.key << std::endl;

    //     // std::cout << "build build_root  " << build_root << std::endl;
    //     // std::cout << "build root  " << root << std::endl;
    //     return build_root;
    // }

    // Node* build_tree_bulk_fmcd(T* _keys, P* _values, int _size)
    // {
    //     RT_ASSERT(_size > 1);

    //     typedef struct {
    //         int begin;
    //         int end;
    //         int level; // top level = 1
    //         Node* node;
    //     } Segment;
    //     std::stack<Segment> s;

    //     Node* ret = new_nodes(1);
    //     s.push((Segment){0, _size, 1, ret});

    //     // int total_level = 0;  // 记录所有层级的总和
    //     // int node_count = 0;   // 记录节点的总数
    //     // int direct_store_count = 0; // 记录第一层直接存储的 Key 数
    //     int confict_num = 0;
    //     int leaf_node_keys_num = 0;
    //     int conflict_leaf_node_keys_num = 0;

    //                                                                     //这个下面的代码是看下alex每个叶子节点里面的键数量
    //                                                                     // std::vector<T> first_keys;
    //                                                                     // std::ifstream infile("alex_libio_first_keys.txt");    // 打开文件

    //                                                                     // if (!infile) {
    //                                                                     //     std::cerr << "Unable to open file: " << "alex_first_keys.txt" << std::endl;
    //                                                                     // }

    //                                                                     // T key;
    //                                                                     // while (infile >> key) {            // 从文件中逐个读取 KEY_TYPE 值
    //                                                                     //     // std::cout << "Read key: " << key << std::endl;
    //                                                                     //     first_keys.push_back(key);     // 将读取到的值存入 vector
    //                                                                     // }

    //                                                                     // infile.close();                    // 关闭文件
    //                                                                     // // std::cout << "first_keys.size: " << first_keys.size() << std::endl;
    //                                                                     // std::vector<int> counts(first_keys.size(), 0); // 初始化计数数组
    //                                                                     // int j = 0; // 用于遍历 first_keys
    //                                                                     // int count = 0;

    //                                                                     // std::ofstream outfile("alex_libio_datanode_keynum");
    //                                                                     // if (!outfile) {
    //                                                                     //     std::cerr << "Unable to open file: " << "alex_libio_datanode_keynum" << std::endl;
    //                                                                     //     // return;
    //                                                                     // }

    //                                                                     // // 遍历 `_keys` 数组
    //                                                                     // for (int i = 0; i < _size; ++i) {
    //                                                                     //     // 如果当前 `_keys[i]` 超过当前的 `first_keys[j+1]`，则说明进入下一个区间
    //                                                                     //     while (j + 1 < first_keys.size() && _keys[i] >= first_keys[j + 1]) {
    //                                                                     //         outfile << count << std::endl;  // 将当前区间的计数写入文件
    //                                                                     //         count = 0;  // 重置计数
    //                                                                     //         ++j;  // 移动到下一个区间
    //                                                                     //     }

    //                                                                     //     // 统计在当前区间内的 `_keys`
    //                                                                     //     if (_keys[i] >= first_keys[j] && (j + 1 >= first_keys.size() || _keys[i] < first_keys[j + 1])) {
    //                                                                     //         ++count;
    //                                                                     //     }
    //                                                                     // }

    //                                                                     // // 记录最后一个 `first_key` 到 `_keys` 末尾的计数
    //                                                                     // outfile << count << std::endl;

    //                                                                     // outfile.close();
    //                                                                     // std::cout << "Counts have been written to " << "alex_libio_datanode_keynum" << std::endl;
    //                                                                     //这个上面的代码是看下alex每个叶子节点里面的键数量


    //     while (!s.empty()) {
    //         // std::cout << "应该只输出一次 " << std::endl;
    //         const int begin = s.top().begin;
    //         const int end = s.top().end;
    //         const int level = s.top().level;
    //         Node* node = s.top().node;
    //         s.pop();

    //         // // 累积层级并增加节点总数
    //         // total_level += level;
    //         // node_count++;

    //         RT_ASSERT(end - begin >= 2);
    //         // if (end - begin == 2) {
    //         //     std::cout << "不应该进到这里" << std::endl;
    //         //     Node* _ = build_tree_two(_keys[begin], _values[begin], _keys[begin+1], _values[begin+1]);
    //         //     memcpy(node, _, sizeof(Node));
    //         //     delete_nodes(_, 1);
    //         // } else {
    //             T* keys = _keys + begin;
    //             P* values = _values + begin;
    //             const int size = end - begin;
    //             const int BUILD_GAP_CNT = compute_gap_count(size);
    //             // const int BUILD_GAP_CNT = 0;
    //             // std::cout << "size: " << size << std::endl;
    //             // std::cout << "size * static_cast<int>(BUILD_GAP_CNT + 1): " << size * static_cast<int>(BUILD_GAP_CNT + 1) << std::endl;
    //             const int max_items = 20000000; // 尝试将num_items控制的最大阈值为 20000000 120000

    //             node->is_two = 0;
    //             node->build_size = size;
    //             node->size = size;
    //             node->fixed = 0;
    //             node->num_inserts = node->num_insert_to_data = 0;

    //             // FMCD method
    //             // Here the implementation is a little different with Algorithm 1 in our paper.
    //             // In Algorithm 1, U_T should be (keys[size-1-D] - keys[D]) / (L - 2).
    //             // But according to the derivation described in our paper, M.A should be less than 1 / U_T.
    //             // So we added a small number (1e-6) to U_T.
    //             // In fact, it has only a negligible impact of the performance.
    //             {
    //                 const int L = size * static_cast<int>(BUILD_GAP_CNT + 1);
    //                 int i = 0;
    //                 int D = 1;
    //                 RT_ASSERT(D <= size-1-D);
    //                 double Ut = (static_cast<long double>(keys[size - 1 - D]) - static_cast<long double>(keys[D])) /
    //                             (static_cast<double>(L - 2)) + 1e-6;
    //                 while (i < size - 1 - D) {
    //                     while (i + D < size && keys[i + D] - keys[i] >= Ut) {
    //                         i ++;
    //                     }
    //                     if (i + D >= size) {
    //                         break;
    //                     }
    //                     D = D + 1;
    //                     if (D * 3 > size) break;
    //                     RT_ASSERT(D <= size-1-D);
    //                     Ut = (static_cast<long double>(keys[size - 1 - D]) - static_cast<long double>(keys[D])) /
    //                          (static_cast<double>(L - 2)) + 1e-6;
    //                 }
    //                 if (D * 3 <= size) {
    //                     stats.fmcd_success_times ++;

    //                     node->model.a = 1.0 / Ut;
    //                     node->model.b = (L - node->model.a * (static_cast<long double>(keys[size - 1 - D]) +
    //                                                           static_cast<long double>(keys[D]))) / 2;
    //                     RT_ASSERT(isfinite(node->model.a));
    //                     RT_ASSERT(isfinite(node->model.b));
    //                     node->num_items = L;
    //                 } else {
    //                     stats.fmcd_broken_times ++;

    //                     int mid1_pos = (size - 1) / 3;
    //                     int mid2_pos = (size - 1) * 2 / 3;

    //                     RT_ASSERT(0 <= mid1_pos);
    //                     RT_ASSERT(mid1_pos < mid2_pos);
    //                     RT_ASSERT(mid2_pos < size - 1);

    //                     const long double mid1_key = (static_cast<long double>(keys[mid1_pos]) +
    //                                                   static_cast<long double>(keys[mid1_pos + 1])) / 2;
    //                     const long double mid2_key = (static_cast<long double>(keys[mid2_pos]) +
    //                                                   static_cast<long double>(keys[mid2_pos + 1])) / 2;

    //                     node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
    //                     const double mid1_target = mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;
    //                     const double mid2_target = mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;

    //                     node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
    //                     node->model.b = mid1_target - node->model.a * mid1_key;
    //                     RT_ASSERT(isfinite(node->model.a));
    //                     RT_ASSERT(isfinite(node->model.b));
    //                 }
    //             }
    //             RT_ASSERT(node->model.a >= 0);
    //             const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
    //             // std::cout << "lr_remains: " << lr_remains << std::endl;
    //             node->model.b += lr_remains;
    //             node->num_items += lr_remains * 2;

    //             // 调整 node->num_items 以适应阈值，并对模型进行比例调整
    //             if (node->num_items > max_items) {
    //                 double scale_factor = static_cast<double>(max_items) / node->num_items;
                    
    //                 // 对模型的斜率和截距进行比例调整
    //                 node->model.a *= scale_factor;
    //                 node->model.b *= scale_factor;

    //                 // 将 num_items 设为最大阈值
    //                 node->num_items = max_items;
    //             }

    //             // std::cout << "调整后的num_items: " << node->num_items << std::endl;

    //             if (size > 1e6) {
    //                 node->fixed = 1;
    //             }

    //             node->items = new_items(node->num_items);
    //             const int bitmap_size = BITMAP_SIZE(node->num_items);
    //             node->none_bitmap = new_bitmap(bitmap_size);
    //             node->child_bitmap = new_bitmap(bitmap_size);
    //             memset(node->none_bitmap, 0xff, sizeof(bitmap_t) * bitmap_size);
    //             memset(node->child_bitmap, 0, sizeof(bitmap_t) * bitmap_size);




    //             for (int item_i = PREDICT_POS(node, keys[0]), offset = 0; offset < size; ) {
    //                 int next = offset + 1, next_i = -1;
    //                 while (next < size) {
    //                     next_i = PREDICT_POS(node, keys[next]);
    //                     if (next_i == item_i) {
    //                         next ++;
    //                     } else {
    //                         break;
    //                     }
    //                 }

                        
    //                 BITMAP_CLEAR(node->none_bitmap, item_i);
    //                 // BITMAP_SET(node->child_bitmap, item_i);

    //                 const int num_keys = next - offset;
    //                 V* value_pairs = new V[num_keys];

    //                 // 填充数组，将 keys 和 values 转换为 std::pair<T, P>
    //                 for (int i = 0; i < num_keys; ++i) {
    //                     value_pairs[i] = std::make_pair(_keys[begin + offset + i], _values[begin + offset + i]);
    //                 }

    //                 auto data_node = new (data_node_allocator().allocate(1))
    //                     alex::AlexDataNode<T, P>(level + 1, derived_params_.max_data_node_slots,
    //                                     key_less_, allocator_);
    //                 confict_num ++;
    //                 data_node->bulk_load(value_pairs, num_keys);
    //                 node->items[item_i].comp.leaf_node = data_node;
    //                 leaf_node_keys_num += num_keys;
    //                 conflict_leaf_node_keys_num += num_keys;  

    //                 if (next >= size) {
    //                     break;
    //                 } else {
    //                     item_i = next_i;
    //                     offset = next;
    //                 }
    //             }


    //             // int total_keys_in_current_node = 0; // 当前 data_node 的累计键值对数量
    //             // alex::AlexDataNode<T, P>* current_data_node = nullptr; // 当前 data_node
    //             // std::vector<std::pair<T, P>> pending_pairs; // 用于暂存未达到3000数量的键值对
    //             // std::vector<int> pending_item_indices; // 存储待指向 data_node 的 item 索引

    //             // for (int item_i = PREDICT_POS(node, keys[0]), offset = 0; offset < size;) {
    //             //     int next = offset + 1, next_i = -1;

    //             //     // 处理同一位置的冲突情况
    //             //     while (next < size) {
    //             //         next_i = PREDICT_POS(node, keys[next]);
    //             //         if (next_i == item_i) {
    //             //             next++;
    //             //         } else {
    //             //             break;
    //             //         }
    //             //     }

    //             //     BITMAP_CLEAR(node->none_bitmap, item_i);

    //             //     const int num_keys = next - offset;
    //             //     total_keys_in_current_node += num_keys;

    //             //     // 将当前的 keys 和 values 转换为 std::pair<T, P> 并添加到暂存向量
    //             //     for (int i = 0; i < num_keys; ++i) {
    //             //         pending_pairs.emplace_back(_keys[begin + offset + i], _values[begin + offset + i]);
    //             //     }

    //             //     // 暂存需要更新指针的位置
    //             //     pending_item_indices.push_back(item_i);

    //             //     // 当累计键值对数量超过3000时，创建新的 data_node 并批量加载
    //             //     if (total_keys_in_current_node > 3000) {
    //             //         current_data_node = new (data_node_allocator().allocate(1)) alex::AlexDataNode<T, P>(
    //             //             level + 1, derived_params_.max_data_node_slots, key_less_, allocator_);
    //             //         confict_num++;

    //             //         // 加载数据到新创建的 data_node
    //             //         current_data_node->bulk_load(pending_pairs.data(), pending_pairs.size());

    //             //         // 将所有待更新位置的指针指向当前的 data_node
    //             //         for (int index : pending_item_indices) {
    //             //             node->items[index].comp.leaf_node = current_data_node;
    //             //         }
    //             //         leaf_node_keys_num += total_keys_in_current_node;
    //             //         conflict_leaf_node_keys_num += total_keys_in_current_node;

    //             //         // 清空暂存向量和索引，重置计数器
    //             //         pending_pairs.clear();
    //             //         pending_item_indices.clear();
    //             //         total_keys_in_current_node = 0;
    //             //     }



    //             //     // 如果已到达末尾，退出循环
    //             //     if (next >= size) {
    //             //         break;
    //             //     } else {
    //             //         item_i = next_i;
    //             //         offset = next;
    //             //     }
    //             // }

    //             // // 最后处理不足3000个的剩余键值对
    //             // if (!pending_pairs.empty()) {
    //             //     current_data_node = new (data_node_allocator().allocate(1)) alex::AlexDataNode<T, P>(
    //             //         level + 1, derived_params_.max_data_node_slots, key_less_, allocator_);
    //             //     confict_num++;

    //             //     // 加载剩余数据
    //             //     current_data_node->bulk_load(pending_pairs.data(), pending_pairs.size());

    //             //     leaf_node_keys_num += pending_pairs.size();
    //             //     conflict_leaf_node_keys_num += pending_pairs.size();

    //             //     // 将所有剩余的待更新位置的指针指向最后的 data_node
    //             //     for (int index : pending_item_indices) {
    //             //         node->items[index].comp.leaf_node = current_data_node;
    //             //     }

    //             //     pending_pairs.clear();
    //             // }

    //                                                                             // for (int item_i = PREDICT_POS(node, keys[0]), offset = 0; offset < size; ) {
    //                                                                             //     int next = offset + 1, next_i = -1;
    //                                                                             //     while (next < size) {
    //                                                                             //         next_i = PREDICT_POS(node, keys[next]);
    //                                                                             //         if (next_i == item_i) {
    //                                                                             //             next ++;
    //                                                                             //         } else {
    //                                                                             //             break;
    //                                                                             //         }
    //                                                                             //     }
    //                                                                             //     //下面这俩其实可以合并起来的 这个if else没有存在的必要
    //                                                                             //     // if (next == offset + 1) {
    //                                                                             //     //     // BITMAP_CLEAR(node->none_bitmap, item_i);
    //                                                                             //     //     // node->items[item_i].comp.data.key = keys[offset];
    //                                                                             //     //     // node->items[item_i].comp.data.value = values[offset];
    //                                                                             //     //     // // std::cout << "只有一个节点 精确预测: " << keys[offset] << std::endl;

    //                                                                             //     //     // confict_num ++;
    //                                                                             //     //     BITMAP_CLEAR(node->none_bitmap, item_i);
    //                                                                             //     //     // BITMAP_SET(node->child_bitmap, item_i);
    //                                                                             //     //     // node->items[item_i].comp.child = new_nodes(1);
    //                                                                             //     //     V* value_pairs = new V[1];
    //                                                                             //     //     value_pairs[0] = std::make_pair(keys[offset], values[offset]);
    //                                                                             //     //     auto data_node = new (data_node_allocator().allocate(1))
    //                                                                             //     //         alex::AlexDataNode<T, P>(level + 1, derived_params_.max_data_node_slots,
    //                                                                             //     //                         key_less_, allocator_);
    //                                                                             //     //     data_node->bulk_load(value_pairs, 1);
    //                                                                             //     //     node->items[item_i].comp.leaf_node = data_node;
    //                                                                             //     //     leaf_node_keys_num += 1;         
    //                                                                             //     // } else {
                                                                                        
    //                                                                             //         BITMAP_CLEAR(node->none_bitmap, item_i);
    //                                                                             //         // BITMAP_SET(node->child_bitmap, item_i);

    //                                                                             //         const int num_keys = next - offset;
    //                                                                             //         V* value_pairs = new V[num_keys];

    //                                                                             //         // 填充数组，将 keys 和 values 转换为 std::pair<T, P>
    //                                                                             //         for (int i = 0; i < num_keys; ++i) {
    //                                                                             //             value_pairs[i] = std::make_pair(_keys[begin + offset + i], _values[begin + offset + i]);
    //                                                                             //         }

    //                                                                             //         auto data_node = new (data_node_allocator().allocate(1))
    //                                                                             //             alex::AlexDataNode<T, P>(level + 1, derived_params_.max_data_node_slots,
    //                                                                             //                             key_less_, allocator_);
    //                                                                             //         confict_num ++;
    //                                                                             //         data_node->bulk_load(value_pairs, num_keys);
    //                                                                             //         node->items[item_i].comp.leaf_node = data_node;
    //                                                                             //         // node->items[item_i].item_model.a_ = data_node->model_.a_;
    //                                                                             //         // node->items[item_i].item_model.b_ = data_node->model_.b_;
    //                                                                             //         leaf_node_keys_num += num_keys;
    //                                                                             //         conflict_leaf_node_keys_num += num_keys;                     
    //                                                                             //     // }
    //                                                                             //     if (next >= size) {
    //                                                                             //         break;
    //                                                                             //     } else {
    //                                                                             //         item_i = next_i;
    //                                                                             //         offset = next;
    //                                                                             //     }
    //                                                                             // }
    //         // }
    //     }
    //     std::cout << "confict_num : " << confict_num << "  conflict_leaf_node_keys_num: " << conflict_leaf_node_keys_num << std::endl;
    //     std::cout << "leaf_node_keys_num: " << leaf_node_keys_num << std::endl;
    //     return ret;
    // }


    // 计算了costmodel的bulk_load方法
    // Node* build_tree_bulk_fmcd(T* _keys, P* _values, int _size)
    // {
    //     RT_ASSERT(_size > 1);

    //     typedef struct {
    //         int begin;
    //         int end;
    //         int level; // top level = 1
    //         Node* node;
    //     } Segment;
    //     std::stack<Segment> s;

    //     Node* ret = new_nodes(1);
    //     s.push((Segment){0, _size, 1, ret});

    //     // int total_level = 0;  // 记录所有层级的总和
    //     // int node_count = 0;   // 记录节点的总数
    //     // int direct_store_count = 0; // 记录第一层直接存储的 Key 数
    //     int confict_num = 0;
    //     int leaf_node_keys_num = 0;

    //     while (!s.empty()) {
    //         // std::cout << "应该只输出一次 " << std::endl;
    //         const int begin = s.top().begin;
    //         const int end = s.top().end;
    //         const int level = s.top().level;
    //         Node* node = s.top().node;
    //         s.pop();

    //         // // 累积层级并增加节点总数
    //         // total_level += level;
    //         // node_count++;

    //         RT_ASSERT(end - begin >= 2);
    //         if (end - begin == 2) {
    //             std::cout << "不应该进到这里" << std::endl;
    //             Node* _ = build_tree_two(_keys[begin], _values[begin], _keys[begin+1], _values[begin+1]);
    //             memcpy(node, _, sizeof(Node));
    //             delete_nodes(_, 1);
    //         } else {
    //             T* keys = _keys + begin;
    //             P* values = _values + begin;
    //             const int size = end - begin;
    //             const int BUILD_GAP_CNT = compute_gap_count(size);
    //             // const int BUILD_GAP_CNT = 0;
    //             // std::cout << "size: " << size << std::endl;
    //             // std::cout << "size * static_cast<int>(BUILD_GAP_CNT + 1): " << size * static_cast<int>(BUILD_GAP_CNT + 1) << std::endl;
    //             const int max_items = 10000000; // 尝试将num_items控制的最大阈值为 20000000

    //             node->is_two = 0;
    //             node->build_size = size;
    //             node->size = size;
    //             node->fixed = 0;
    //             node->num_inserts = node->num_insert_to_data = 0;

    //             // FMCD method
    //             // Here the implementation is a little different with Algorithm 1 in our paper.
    //             // In Algorithm 1, U_T should be (keys[size-1-D] - keys[D]) / (L - 2).
    //             // But according to the derivation described in our paper, M.A should be less than 1 / U_T.
    //             // So we added a small number (1e-6) to U_T.
    //             // In fact, it has only a negligible impact of the performance.
    //             {
    //                 const int L = size * static_cast<int>(BUILD_GAP_CNT + 1);
    //                 int i = 0;
    //                 int D = 1;
    //                 RT_ASSERT(D <= size-1-D);
    //                 double Ut = (static_cast<long double>(keys[size - 1 - D]) - static_cast<long double>(keys[D])) /
    //                             (static_cast<double>(L - 2)) + 1e-6;
    //                 while (i < size - 1 - D) {
    //                     while (i + D < size && keys[i + D] - keys[i] >= Ut) {
    //                         i ++;
    //                     }
    //                     if (i + D >= size) {
    //                         break;
    //                     }
    //                     D = D + 1;
    //                     if (D * 3 > size) break;
    //                     RT_ASSERT(D <= size-1-D);
    //                     Ut = (static_cast<long double>(keys[size - 1 - D]) - static_cast<long double>(keys[D])) /
    //                          (static_cast<double>(L - 2)) + 1e-6;
    //                 }
    //                 if (D * 3 <= size) {
    //                     stats.fmcd_success_times ++;

    //                     node->model.a = 1.0 / Ut;
    //                     node->model.b = (L - node->model.a * (static_cast<long double>(keys[size - 1 - D]) +
    //                                                           static_cast<long double>(keys[D]))) / 2;
    //                     RT_ASSERT(isfinite(node->model.a));
    //                     RT_ASSERT(isfinite(node->model.b));
    //                     node->num_items = L;
    //                 } else {
    //                     stats.fmcd_broken_times ++;

    //                     int mid1_pos = (size - 1) / 3;
    //                     int mid2_pos = (size - 1) * 2 / 3;

    //                     RT_ASSERT(0 <= mid1_pos);
    //                     RT_ASSERT(mid1_pos < mid2_pos);
    //                     RT_ASSERT(mid2_pos < size - 1);

    //                     const long double mid1_key = (static_cast<long double>(keys[mid1_pos]) +
    //                                                   static_cast<long double>(keys[mid1_pos + 1])) / 2;
    //                     const long double mid2_key = (static_cast<long double>(keys[mid2_pos]) +
    //                                                   static_cast<long double>(keys[mid2_pos + 1])) / 2;

    //                     node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
    //                     const double mid1_target = mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;
    //                     const double mid2_target = mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) + static_cast<int>(BUILD_GAP_CNT + 1) / 2;

    //                     node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
    //                     node->model.b = mid1_target - node->model.a * mid1_key;
    //                     RT_ASSERT(isfinite(node->model.a));
    //                     RT_ASSERT(isfinite(node->model.b));
    //                 }
    //             }
    //             RT_ASSERT(node->model.a >= 0);
    //             const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
    //             // std::cout << "lr_remains: " << lr_remains << std::endl;
    //             node->model.b += lr_remains;
    //             node->num_items += lr_remains * 2;

    //             // 调整 node->num_items 以适应阈值，并对模型进行比例调整
    //             if (node->num_items > max_items) {
    //                 double scale_factor = static_cast<double>(max_items) / node->num_items;
                    
    //                 // 对模型的斜率和截距进行比例调整
    //                 node->model.a *= scale_factor;
    //                 node->model.b *= scale_factor;

    //                 // 将 num_items 设为最大阈值
    //                 node->num_items = max_items;
    //             }

    //             // std::cout << "调整后的num_items: " << node->num_items << std::endl;

    //             if (size > 1e6) {
    //                 node->fixed = 1;
    //             }

    //             node->items = new_items(node->num_items);
    //             const int bitmap_size = BITMAP_SIZE(node->num_items);
    //             node->none_bitmap = new_bitmap(bitmap_size);
    //             node->child_bitmap = new_bitmap(bitmap_size);
    //             memset(node->none_bitmap, 0xff, sizeof(bitmap_t) * bitmap_size);
    //             memset(node->child_bitmap, 0, sizeof(bitmap_t) * bitmap_size);

    //             for (int item_i = PREDICT_POS(node, keys[0]), offset = 0; offset < size; ) {
    //                 int next = offset + 1, next_i = -1;
    //                 while (next < size) {
    //                     next_i = PREDICT_POS(node, keys[next]);
    //                     if (next_i == item_i) {
    //                         next ++;
    //                     } else {
    //                         break;
    //                     }
    //                 }
    //                 if (next == offset + 1) {
    //                     // BITMAP_CLEAR(node->none_bitmap, item_i);
    //                     // node->items[item_i].comp.data.key = keys[offset];
    //                     // node->items[item_i].comp.data.value = values[offset];
    //                     // // std::cout << "只有一个节点 精确预测: " << keys[offset] << std::endl;
    //                     confict_num ++;
    //                     BITMAP_CLEAR(node->none_bitmap, item_i);
    //                     // BITMAP_SET(node->child_bitmap, item_i);
    //                     node->items[item_i].comp.child = new_nodes(1);
    //                     V* value_pairs = new V[1];
    //                     value_pairs[0] = std::make_pair(keys[offset], values[offset]);
    //                     auto data_node = new (data_node_allocator().allocate(1))
    //                         alex::AlexDataNode<T, P>(level + 1, derived_params_.max_data_node_slots,
    //                                         key_less_, allocator_);
    //                     data_node->bulk_load(value_pairs, 1);
    //                     node->items[item_i].comp.leaf_node = data_node;
    //                     leaf_node_keys_num += 1;         
    //                 } else {
                        
    //                     BITMAP_CLEAR(node->none_bitmap, item_i);
    //                     // BITMAP_SET(node->child_bitmap, item_i);
    //                     // node->items[item_i].comp.child = new_nodes(1);


    //                     // Node* nodenode = node;
    //                     // double cost2 = compute_leaf_cost(nodenode, keys, values, next - offset);



    //                     const int num_keys = next - offset;
    //                     V* value_pairs = new V[num_keys];

    //                     // 填充数组，将 keys 和 values 转换为 std::pair<T, P>
    //                     for (int i = 0; i < num_keys; ++i) {
    //                         value_pairs[i] = std::make_pair(_keys[begin + offset + i], _values[begin + offset + i]);
    //                     }

    //                     auto data_node = new (data_node_allocator().allocate(1))
    //                         alex::AlexDataNode<T, P>(level + 1, derived_params_.max_data_node_slots,
    //                                         key_less_, allocator_);

    //                     if(num_keys > 2){
    //                         alex::LinearModel<T> data_node_model;
    //                         alex::AlexDataNode<T,P>::build_model(value_pairs, num_keys, &data_node_model, params_.approximate_model_computation);
    //                         alex::DataNodeStats stats;
    //                         data_node->cost_ = alex::AlexDataNode<T,P>::compute_expected_cost(
    //                             value_pairs, num_keys, alex::AlexDataNode<T,P>::kInitDensity_,
    //                             params_.expected_insert_frac, &data_node_model,
    //                             params_.approximate_cost_computation, &stats);
    //                         double cost2 = data_node->cost_;
    //                         // std::cout << "直接把冲突数据装进alexdatanode里的cost: " << cost2 << std::endl;

    //                         Node nodenode2 = *node;
    //                         double cost1 = compute_cost(&nodenode2, _keys, _values, begin + offset, begin + next) + 20;//20就相当于是traversetoleaf的cost
    //                         // std::cout << "多加一层lipp之后 alexdatanode里的cost 不算多的那层lipp的cost和traversetoleaf的cost: " << cost1 << std::endl;

    //                         if(cost2 <= cost1){
    //                             confict_num ++;
    //                             data_node->bulk_load(value_pairs, num_keys);
    //                             node->items[item_i].comp.leaf_node = data_node;
    //                             leaf_node_keys_num += num_keys;                                
    //                         } else {
    //                             // std::cout << "cost2: " << cost2 << "cost1: " << cost1 << std::endl;
    //                             BITMAP_SET(node->child_bitmap, item_i);
    //                             node->items[item_i].comp.child = new_nodes(1);
    //                             s.push((Segment){begin + offset, begin + next, level + 1, node->items[item_i].comp.child});
    //                         }
    //                     } else {
    //                         confict_num ++;
    //                         data_node->bulk_load(value_pairs, num_keys);
    //                         node->items[item_i].comp.leaf_node = data_node;
    //                         leaf_node_keys_num += num_keys;                          
    //                     }



    //                     // if(level == 0){
    //                     //     BITMAP_SET(node->child_bitmap, item_i);
    //                     //     node->items[item_i].comp.child = new_nodes(1);
    //                     //     // Node* nodenode2 = node;
    //                     //     // double cost1 = compute_cost(nodenode2, _keys, _values, begin + offset, begin + next);
    //                     //     // std::cout << "多加一层lipp之后 alexdatanode里的cost 不算多的那层lipp的cost和traversetoleaf的cost: " << cost1 << std::endl;
    //                     //     s.push((Segment){begin + offset, begin + next, level + 1, node->items[item_i].comp.child});
    //                     // } else {
    //                     //     const int num_keys = next - offset;
    //                     //     V* value_pairs = new V[num_keys];

    //                     //     // 填充数组，将 keys 和 values 转换为 std::pair<T, P>
    //                     //     for (int i = 0; i < num_keys; ++i) {
    //                     //         value_pairs[i] = std::make_pair(_keys[begin + offset + i], _values[begin + offset + i]);
    //                     //     }

    //                     //     auto data_node = new (data_node_allocator().allocate(1))
    //                     //         alex::AlexDataNode<T, P>(level + 1, derived_params_.max_data_node_slots,
    //                     //                         key_less_, allocator_);
    //                     //     data_node->bulk_load(value_pairs, num_keys);
    //                     //     node->items[item_i].comp.leaf_node = data_node;
    //                     //     leaf_node_keys_num += num_keys;
    //                     // }


    //                     // s.push((Segment){begin + offset, begin + next, level + 1, node->items[item_i].comp.child});

    //                     // alex::LinearModel<T> data_node_model;
    //                     // alex::AlexDataNode<T,P>::build_model(value_pairs, num_keys, &data_node_model, params_.approximate_model_computation);
    //                     // alex::DataNodeStats stats;
    //                     // data_node->cost_ = alex::AlexDataNode<T,P>::compute_expected_cost(
    //                     //     value_pairs, num_keys, alex::AlexDataNode<T,P>::kInitDensity_,
    //                     //     params_.expected_insert_frac, &data_node_model,
    //                     //     params_.approximate_cost_computation, &stats);

    //                     // std::cout << "看一下datanode_cost的数量级: " << data_node->cost_ << std::endl;  //基本都在个位数
                        
    //                 }
    //                 if (next >= size) {
    //                     break;
    //                 } else {
    //                     item_i = next_i;
    //                     offset = next;
    //                 }
    //             }
    //         }
    //     }
    //     std::cout << "confict_num也就是alexdatanode的数量: " << confict_num << std::endl;
    //     std::cout << "leaf_node_keys_num: " << leaf_node_keys_num << std::endl;
    //     // // 打印第一层可以直接存储的 Key 数量
    //     // printf("第一层拟合中可以直接存储的 Key 数量: %d\n", direct_store_count);

    //     // // 计算平均层级（树的平均高度）
    //     // double average_level = static_cast<double>(total_level) / node_count;
    //     // std::cout << "构建树的平均层级: " << average_level << std::endl;

    //     return ret;
    // }

    void destory_pending()
    {
        while (!pending_two.empty()) {
            Node* node = pending_two.top(); pending_two.pop();

            delete_items(node->items, node->num_items);
            const int bitmap_size = BITMAP_SIZE(node->num_items);
            delete_bitmap(node->none_bitmap, bitmap_size);
            delete_bitmap(node->child_bitmap, bitmap_size);
            delete_nodes(node, 1);
        }
    }

    void destroy_tree(Node* root)
    {
        std::stack<Node*> s;
        s.push(root);
        while (!s.empty()) {
            Node* node = s.top(); s.pop();

            for (int i = 0; i < node->num_items; i ++) {
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    s.push(node->items[i].comp.child);
                }
            }

            if (node->is_two) {
                RT_ASSERT(node->build_size == 2);
                RT_ASSERT(node->num_items == 8);
                node->size = 2;
                node->num_inserts = node->num_insert_to_data = 0;
                node->none_bitmap[0] = 0xff;
                node->child_bitmap[0] = 0;
                pending_two.push(node);
            } else {
                delete_items(node->items, node->num_items);
                const int bitmap_size = BITMAP_SIZE(node->num_items);
                delete_bitmap(node->none_bitmap, bitmap_size);
                delete_bitmap(node->child_bitmap, bitmap_size);
                delete_nodes(node, 1);
            }
        }
    }

    void scan_and_destory_tree(Node* _root, T* keys, P* values, bool destory = true)
    {
        typedef std::pair<int, Node*> Segment; // <begin, Node*>
        std::stack<Segment> s;

        s.push(Segment(0, _root));
        while (!s.empty()) {
            int begin = s.top().first;
            Node* node = s.top().second;
            const int SHOULD_END_POS = begin + node->size;
            s.pop();

            for (int i = 0; i < node->num_items; i ++) {
                if (BITMAP_GET(node->none_bitmap, i) == 0) {
                    if (BITMAP_GET(node->child_bitmap, i) == 0) {
                        //11111111111111111111111111111111111111111111111111
                        // keys[begin] = node->items[i].comp.data.key;
                        // values[begin] = node->items[i].comp.data.value;
                        begin ++;
                    } else {
                        s.push(Segment(begin, node->items[i].comp.child));
                        begin += node->items[i].comp.child->size;
                    }
                }
            }
            RT_ASSERT(SHOULD_END_POS == begin);

            if (destory) {
                if (node->is_two) {
                    RT_ASSERT(node->build_size == 2);
                    RT_ASSERT(node->num_items == 8);
                    node->size = 2;
                    node->num_inserts = node->num_insert_to_data = 0;
                    node->none_bitmap[0] = 0xff;
                    node->child_bitmap[0] = 0;
                    pending_two.push(node);
                } else {
                    delete_items(node->items, node->num_items);
                    const int bitmap_size = BITMAP_SIZE(node->num_items);
                    delete_bitmap(node->none_bitmap, bitmap_size);
                    delete_bitmap(node->child_bitmap, bitmap_size);
                    delete_nodes(node, 1);
                }
            }
        }
    }

    Node* insert_tree(Node* _node, const T& key, const P& value, bool* ok = nullptr)
    {
        constexpr int MAX_DEPTH = 128;
        Node* path[MAX_DEPTH];
        int path_size = 0;
        int insert_to_data = 0;

        for (Node* node = _node; ; ) {
            RT_ASSERT(path_size < MAX_DEPTH);
            path[path_size ++] = node;

            node->size ++;
            node->num_inserts ++;
            int pos = PREDICT_POS(node, key);
                
            if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                //这里是插到gap位置了，直接把这个键值对bulk_load进一个alexdatanode 记得设置bitmap
                BITMAP_CLEAR(node->none_bitmap, pos);                
                std::pair<T, P> value_pair{key, value};
                auto data_node = new (data_node_allocator().allocate(1))
                    alex::AlexDataNode<T, P>(1, derived_params_.max_data_node_slots, key_less_, allocator_);
                data_node->bulk_load(&value_pair, 1);
                node->items[pos].comp.leaf_node = data_node;
                // BITMAP_CLEAR(node->child_bitmap, pos); //data_node->bulk_load会营销到child_bitmap
                break;
            } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
                auto alexnode = node->items[pos].comp.leaf_node;
                std::pair<int, int> ret = alexnode->insert(key, value);
                int fail = ret.first;
                int insert_pos = ret.second;
                if (fail == -1) {
                    // Duplicate found and duplicates not allowed
                    *ok = false;
                    break;
                }

                if (fail) {
                    //如果第一次失败了，直接尝试一下扩大斜率，重新训练模型，如果还是不行，直接split down
                    //实验的时候，看一下在这部分扩大斜率之后能够成功插入的有多少，看下划不划算
                    alexnode->resize(alex::AlexDataNode<T, P>::kMinDensity_, true,
                                alexnode->is_append_mostly_right(),
                                alexnode->is_append_mostly_left());
                    // fanout_tree::FTNode& tree_node = used_fanout_tree_nodes[0];
                    // leaf->cost_ = tree_node.cost;
                    //一会把下面这些都注释掉试试 ，好像对我现在的结构来说 没有什么意义
                    alexnode->expected_avg_exp_search_iterations_ = 0;
                    alexnode->expected_avg_shifts_ = 0;
                    alexnode->reset_stats();
                    // Try again to insert the key
                    ret = alexnode->insert(key, value);
                    fail = ret.first;
                    insert_pos = ret.second;
                    if(fail == 0){
                        break;
                    }
                    if (fail == -1) {
                        // Duplicate found and duplicates not allowed
                        *ok = false;
                        break;
                    }
                }
                if(fail){     
                    //把alexdatanode里所有的键拿出来，重新用lipp的fmcd方法来算 或者看下那个fast方法 把新插入的键也加上一起重新算，重新排序
                    int size = alexnode->num_keys_ + 1;
                    T* keys = new T[size];
                    P* values = new P[size];
                    // 提取 keys 和 values 并在递增顺序中插入新键值对
                    alexnode->extract_keys_and_values_with_insertion(keys, values, size, key, value);
                    if (size <= 2) {
                        //bulk load进一个alexdatanode里 然后直接Break就可以
                        //所有指向这个datanode的指针也都得更新
                        std::vector<std::pair<T, P>> value_pairs;
                        // 填充数组，将 keys 和 values 转换为 std::pair<T, P>
                        for (int i = 0; i < size; ++i) {
                            value_pairs.emplace_back(keys[i], values[i]);
                        }
                        auto data_node = new (data_node_allocator().allocate(1))
                            alex::AlexDataNode<T, P>(1, derived_params_.max_data_node_slots,
                                            key_less_, allocator_);
                        data_node->bulk_load(value_pairs.data(), size);
                        for (const auto& [key, value] : value_pairs) {
                            auto [node, item_i] = build_at(root, key);
                            node->items[item_i].comp.leaf_node = data_node;  // 为每个 key 设置 leaf_node
                        }
                        delete_alexdatanode(alexnode);
                    } else {
                       Node* child_node = insert_build_fmcd(keys, values, size);
                        std::set<std::pair<Node*, int>> node_item_set;
                        for (int i = 0; i < size; ++i) {
                            const auto& key = keys[i];
                            auto [node1, item_i] = build_at(root, key);
                            node1->items[item_i].comp.leaf_node = nullptr;////////////注释掉这句话试试
                            node_item_set.emplace(node1, item_i);
                        }
                        // 遍历不重复的 (Node*, int) 组合
                        for (const auto& [node2, item_i] : node_item_set) {
                            BITMAP_SET(node2->child_bitmap, item_i);
                            node2->items[item_i].comp.child = child_node;
                        }
                        delete_alexdatanode(alexnode);
                    }  
                    // 释放临时数组
                    delete[] keys;
                    delete[] values;
                    break;
                }
                // insert_to_data = 1;
                break;
            } else {
                node = node->items[pos].comp.child;
            }
        }
        // for (int i = 0; i < path_size; i ++) {
        //     path[i]->num_insert_to_data += insert_to_data;
        // }

        // for (int i = 0; i < path_size; i ++) {
        //     Node* node = path[i];
        //     const int num_inserts = node->num_inserts;
        //     const int num_insert_to_data = node->num_insert_to_data;
        //     const bool need_rebuild = node->fixed == 0 && node->size >= node->build_size * 2 && node->size >= 64 && num_insert_to_data * 10 >= num_inserts;

        //     if (need_rebuild) {
        //         const int ESIZE = node->size;
        //         T* keys = new T[ESIZE];
        //         P* values = new P[ESIZE];

        //         #if COLLECT_TIME
        //         auto start_time_scan = std::chrono::high_resolution_clock::now();
        //         #endif
        //         scan_and_destory_tree(node, keys, values);
        //         #if COLLECT_TIME
        //         auto end_time_scan = std::chrono::high_resolution_clock::now();
        //         auto duration_scan = end_time_scan - start_time_scan;
        //         stats.time_scan_and_destory_tree += std::chrono::duration_cast<std::chrono::nanoseconds>(duration_scan).count() * 1e-9;
        //         #endif

        //         #if COLLECT_TIME
        //         auto start_time_build = std::chrono::high_resolution_clock::now();
        //         #endif
        //         Node* new_node = build_tree_bulk(keys, values, ESIZE);
        //         #if COLLECT_TIME
        //         auto end_time_build = std::chrono::high_resolution_clock::now();
        //         auto duration_build = end_time_build - start_time_build;
        //         stats.time_build_tree_bulk += std::chrono::duration_cast<std::chrono::nanoseconds>(duration_build).count() * 1e-9;
        //         #endif

        //         delete[] keys;
        //         delete[] values;

        //         path[i] = new_node;
        //         if (i > 0) {
        //             int pos = PREDICT_POS(path[i-1], key);
        //             path[i-1]->items[pos].comp.child = new_node;
        //         }

        //         break;
        //     }
        // }

        // if(xxx == 1 && yyy<=2){
        //     Node * aaa = target_child_node;
        //     std::cout << "target_child_node " << target_child_node << std::endl;
        //     std::cout << "aaa " << aaa << std::endl;
        //     std::cout  << "  aaa->items[0].comp.leaf_node " << aaa->items[0].comp.leaf_node << std::endl;
        // }


        if(ok) {
            *ok = true;
        }
        return path[0];
    }

    // int xxx = 0;
    // int yyy = 0;
    // int temp = 0;
    // Node* target_child_node = nullptr;
    // alex::AlexDataNode<T, P>* alexalex = nullptr;
    // // Node* child_node = NULL;
    // // Node* child_node = new_nodes(1); //把这个挪到insert_tree函数里试试！！！！！！！！！！！！！！！！！！！！！！！！
    // Node* insert_tree(Node* _node, const T& key, const P& value, bool* ok = nullptr)
    // {
    //     constexpr int MAX_DEPTH = 128;
    //     Node* path[MAX_DEPTH];
    //     int path_size = 0;
    //     int insert_to_data = 0;

    //     Node* target_node = nullptr;
    //     // Node* child_node = new_nodes(1);//跟放在外面 一样的bug
    //     // Node* child_node = nullptr;
    //     // std::cout << " key " << key << std::endl;
    //                                 // if(key == 224042607){
    //                                 //     std::cout << " before for target_child_node->items[0].comp.leaf_node " << target_child_node->items[0].comp.leaf_node << std::endl;
    //                                 //     // temp = 1;
    //                                 // }
    //     for (Node* node = _node; ; ) {
    //         RT_ASSERT(path_size < MAX_DEPTH);
    //         path[path_size ++] = node;

    //         node->size ++;
    //         node->num_inserts ++;
    //         int pos = PREDICT_POS(node, key);
    //                 // if(key == 218848377){
    //                 if(key == 224042607){
    //                     std::cout << " root " << root << std::endl;
    //                     std::cout << "49685138node: " << node << "  node->model.a: " << node->model.a << "  node->model.b: " << node->model.b << " pos " << pos << " key " << key << std::endl;
    //                     std::cout << "49685138BITMAP_GET(node->child_bitmap, pos) " << BITMAP_GET(node->child_bitmap, pos) << " BITMAP_GET(node->none_bitmap, pos) " << BITMAP_GET(node->none_bitmap, pos) << std::endl;
    //                     if(BITMAP_GET(node->child_bitmap, pos) == 1){
    //                         std::cout << " ode->items[pos].comp.child " << node->items[pos].comp.child << std::endl;
    //                         auto ccc = node->items[pos].comp.child;
    //                         std::cout << " ccc->items[0].comp.leaf_node " << ccc->items[0].comp.leaf_node << std::endl;
    //                     }
    //                 }
    //                         //         if(std::abs(node->model.a - 0.000239909) < 1e-6 && pos == 52493){
                                        
    //                         //             if (BITMAP_GET(node->child_bitmap, pos) == 1 && BITMAP_GET(node->none_bitmap, pos) == 0){
    //                         //                 std::cout << "bbbbbnode->model.a: " << node->model.a << "  node->model.b: " << node->model.b << "  node: " << node << " key " << key   << " pos " << pos<< std::endl;
    //                         //                 auto child = node->items[pos].comp.child;
    //                         //                 std::cout << "child: " << child << std::endl;
    //                         //                 if (BITMAP_GET(child->child_bitmap, 0) == 0 && BITMAP_GET(child->none_bitmap, pos) == 0){
    //                         //                     auto alexdatanode = child->items[0].comp.leaf_node;
    //                         //                     std::cout << "alexdatanode: " << alexdatanode<< std::endl;
    //                         //                     if(alexdatanode!=alexalex){
    //                         //                         std::cout << "alexalex: " << alexalex<< std::endl;
    //                         //                         exit(1);
    //                         //                     }
    //                         //                 }
    //                         //             }
    //                         //         }

    //                         //         if(std::abs(node->model.a - 0.741327) < 1e-6){
    //                         //             std::cout << "111node->model.a: " << node->model.a << "  node->model.b: " << node->model.b << "  node: " << node << " key " << key   << " pos " << pos<< std::endl;
    //                         //             target_node = node;
    //                         //             // yyy++;
    //                         //             // std::cout << "wwwBITMAP_GET(node->child_bitmap, 0): " << BITMAP_GET(node->child_bitmap, 0)<< std::endl;
    //                         //             // std::cout << "wwwBITMAP_GET(node->none_bitmap, 0): " << BITMAP_GET(node->none_bitmap, 0)<< std::endl;
    //                         //                 if (BITMAP_GET(node->child_bitmap, 0) == 0 && BITMAP_GET(node->none_bitmap, 0) == 0){
    //                         //                     auto alexdatanode = node->items[0].comp.leaf_node;
    //                         //                     auto childnode = node->items[0].comp.child;
    //                         //                     std::cout << "22221alexdatanode: " << alexdatanode << " childnode " << childnode << std::endl;
    //                         //                 }
    //                         //                 if (BITMAP_GET(node->child_bitmap, 1) == 0 && BITMAP_GET(node->none_bitmap, 1) == 0){
    //                         //                     auto alexdatanode = node->items[1].comp.leaf_node;
    //                         //                     auto childnode = node->items[1].comp.child;
    //                         //                     std::cout << "22223alexdatanode: " << alexdatanode << " childnode " << childnode << std::endl;
    //                         //                 }
    //                         //                 if (BITMAP_GET(node->child_bitmap, 12161) == 0 && BITMAP_GET(node->none_bitmap, 12161) == 0){
    //                         //                     auto alexdatanode = node->items[12161].comp.leaf_node;
    //                         //                     auto childnode = node->items[12161].comp.child;
    //                         //                     std::cout << "22224alexdatanode: " << alexdatanode << " childnode " << childnode << std::endl;
    //                         //                 }
    //                         //         }
    //                         //         if(std::abs(node->model.a - 0.741327) < 1e-6 && pos == 0){
    //                         //             std::cout << "要对第0位进行修改: " << " node " << node << " key " << key   << " pos " << pos << std::endl;

    //                         //         }

    //             // if(node == target_node){
    //             //     std::cout << "0node->model.a: " << node->model.a << "  node: " << node << " key " << key   << " pos " << pos << std::endl;
    //             //     std::cout << "  node->items[0].comp.leaf_node " << node->items[0].comp.leaf_node << std::endl; 
    //             // }
    //                                 //             if(key == 224042607){
    //                                 //     std::cout << " before if else target_child_node->items[0].comp.leaf_node " << target_child_node->items[0].comp.leaf_node << std::endl;
    //                                 //     // temp = 1;
    //                                 // }
            
    //         if (BITMAP_GET(node->none_bitmap, pos) == 1) {
    //             // if(temp == 1){
    //             //     std::cout <<" key " << key << std::endl;
    //             // }
    //                             // if(std::abs(node->model.a - 0.741327) < 1e-6){
    //                             //     std::cout << "222node->model.a: " << node->model.a << "  node->model.b: " << node->model.b << "  node: " << node << " key " << key   << " pos " << pos << std::endl;
    //                             // }
    //             // if(node == target_node){
    //             //     std::cout << "0node->model.a: " << node->model.a << "  node: " << node << " key " << key   << " pos " << pos << std::endl;
    //             //     std::cout << "  node->items[0].comp.leaf_node " << node->items[0].comp.leaf_node << std::endl; 
    //             // }
    //                                 // if(pos == 3215){
    //                                 //     std::cout << "  0003215key " << key << " node " << node << " pos " << pos << std::endl;
    //                                 //     std::cout << "  0003215node->items[pos].comp.leaf_node " << node->items[pos].comp.leaf_node << std::endl; 
    //                                 // }
    //             //这里是插到gap位置了，直接把这个键值对bulk_load进一个alexdatanode 记得设置bitmap
    //             BITMAP_CLEAR(node->none_bitmap, pos);
    //             // std::pair<T, P> value_pair = std::make_pair(key, value); //直接初始化列表性能好一点吧
    //             std::pair<T, P> value_pair{key, value};
    //             auto data_node = new (data_node_allocator().allocate(1))
    //                 alex::AlexDataNode<T, P>(1, derived_params_.max_data_node_slots, key_less_, allocator_);
    //             data_node->bulk_load(&value_pair, 1);
    //             node->items[pos].comp.leaf_node = data_node;
    //                             // if(pos == 3215){
    //                             //     std::cout << "  1113215key " << key << " node " << node << " pos " << pos << std::endl;
    //                             //     std::cout << "  1113215node->items[pos].comp.leaf_node " << node->items[pos].comp.leaf_node << std::endl; 
    //                             // }
    //             break;
    //         } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
    //             // if(node == target_node){
    //             //     std::cout << "0node->model.a: " << node->model.a << "  node: " << node << " key " << key   << " pos " << pos << std::endl;
    //             //     std::cout << "  node->items[0].comp.leaf_node " << node->items[0].comp.leaf_node << std::endl; 
    //             // }
    //                                 // if(std::abs(node->model.a - 0.741327) < 1e-6){
    //                                 //     std::cout << "333node->model.a: " << node->model.a << "  node->model.b: " << node->model.b << "  node: " << node << " key " << key   << " pos " << pos << std::endl;
    //                                 // }
    //             // if(node == target_node){
    //             //     std::cout << "0node->model.a: " << node->model.a << "  node: " << node << " key " << key   << " pos " << pos << std::endl;
    //             //     std::cout << "  node->items[0].comp.leaf_node " << node->items[0].comp.leaf_node << std::endl; 
    //             // }
    //             // std::cout << " pos " << pos << " node->model.a " << node->model.a << " node->model.b " << node->model.b << " node " << node << " key " << key << std::endl;
    //             // BITMAP_SET(node->child_bitmap, pos); 
    //             // node->items[pos].comp.child = build_tree_two(key, value, node->items[pos].comp.data.key, node->items[pos].comp.data.value);
    //             //这里是插到data位置了，调用alexdatanode的插入方法，主要还是alex的那些分裂，扩展斜率之类的插入方式，这个位置不一定child_bitmap就是1了，要分情况讨论
    //                                 // if(key == 224042607){
    //                                 //     std::cout << " In else if target_child_node->items[0].comp.leaf_node " << target_child_node->items[0].comp.leaf_node << std::endl;
    //                                 //     // temp = 1;
    //                                 // }
    //             auto alexnode = node->items[pos].comp.leaf_node;
    //                                 // if(key == 218848377){
    //                                 if(key == 224042607){
    //                                     std::cout << " rrrrrrrrrralexnode " << alexnode << std::endl;
    //                                     // temp = 1;
    //                                 }
    //                                 //                 if(key == 224042607){
    //                                 //     std::cout << " after new alexnode target_child_node->items[0].comp.leaf_node " << target_child_node->items[0].comp.leaf_node << std::endl;
    //                                 //     // temp = 1;
    //                                 // }
    //             // std::cout << " alexnode->num_keys_ " << alexnode->num_keys_ << std::endl;
    //             std::pair<int, int> ret = alexnode->insert(key, value);
    //             int fail = ret.first;
    //             int insert_pos = ret.second;
    //             if (fail == -1) {
    //                 // Duplicate found and duplicates not allowed
    //                 *ok = false;
    //                 break;
    //             }
    //                                 // if(key == 224042607){
    //                                 //     std::cout << " after insert in alex target_child_node->items[0].comp.leaf_node " << target_child_node->items[0].comp.leaf_node << std::endl;
    //                                 // }
    //             // if(temp == 1 && fail != 0){
    //             //     std::cout << " fail " << fail << " key " << key << std::endl;
    //             // }
    //             // if(fail != 0){
    //             //     std::cout << " 第一个导致分裂的键 fail " << fail << " key " << key << std::endl;
    //             // }

    //             // if (fail) {
    //             //     //如果第一次失败了，直接尝试一下扩大斜率，重新训练模型，如果还是不行，直接split down
    //             //     //实验的时候，看一下在这部分扩大斜率之后能够成功插入的有多少，看下划不划算
    //             //     alexnode->resize(alex::AlexDataNode<T, P>::kMinDensity_, true,
    //             //                 alexnode->is_append_mostly_right(),
    //             //                 alexnode->is_append_mostly_left());
    //             //     // fanout_tree::FTNode& tree_node = used_fanout_tree_nodes[0];
    //             //     // leaf->cost_ = tree_node.cost;
    //             //     alexnode->expected_avg_exp_search_iterations_ = 0;
    //             //     alexnode->expected_avg_shifts_ = 0;
    //             //     alexnode->reset_stats();
    //             //     // Try again to insert the key
    //             //     ret = alexnode->insert(key, value);
    //             //     fail = ret.first;
    //             //     insert_pos = ret.second;
    //             //     if (fail == -1) {
    //             //         // Duplicate found and duplicates not allowed
    //             //         *ok = false;
    //             //         break;
    //             //     }
    //             //     if(fail == 0){
    //             //         break;
    //             //     }
    //             // }
    //             // std::cout << " !!!!!!!!!!!!!!!!!!!!!!!!!有需要split down的!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "<< std::endl;
    //             if(fail){
    //                 // std::cout << " pos " << pos << std::endl;
    //                                     // if(key == 224042607){
    //                                     //     std::cout << " qqqqqqalex " << alexnode << "  " << alexalex << " node " << node << " pos " << pos << std::endl;
    //                                     // }   
    //                                 // if(key == 224042607){
    //                                 //     std::cout << " in fail target_child_node->items[0].comp.leaf_node " << target_child_node->items[0].comp.leaf_node << " alexnode " << alexnode << std::endl;
    //                                 // }       
    //                 //把alexdatanode里所有的键拿出来，重新用lipp的fmcd方法来算 或者看下那个fast方法 把新插入的键也加上一起重新算，重新排序
    //                 int size = alexnode->num_keys_ + 1;
    //                 T* keys = new T[size];
    //                 P* values = new P[size];
    //                 // std::cout << " 000 " << size << std::endl;
    //                                 // if(key == 224042607){
    //                                 //     std::cout << " before extract target_child_node->items[0].comp.leaf_node " << target_child_node->items[0].comp.leaf_node << " alexnode " << alexnode << " target_child_node " << target_child_node  << std::endl;
    //                                 // }
    //                 // 提取 keys 和 values 并在递增顺序中插入新键值对
    //                 alexnode->extract_keys_and_values_with_insertion(keys, values, size, key, value);
    //                 // std::cout << " 111 " << size << std::endl;
    //                 // if(key == 224042607){
    //                 //     std::cout << " xxxalex " << alexnode << "  " << alexalex << " node " << node << " pos " << pos << std::endl;
    //                 //     for (int i = 0; i < size; ++i) {
    //                 //         std::cout << " xxxalex " << alexnode << " keys[i] " << keys[i] << std::endl;
    //                 //     }
    //                 // } 
    //                                 // if(key == 224042607){
    //                                 //     std::cout << " after extract target_child_node->items[0].comp.leaf_node " << target_child_node->items[0].comp.leaf_node << " alexnode " << alexnode << " target_child_node " << target_child_node   << std::endl;
    //                                 // }
                    
    //                 // std::cout << " 111 " << size << std::endl;
    //                 if (size <= 2) {
    //                     // std::cout << " 222 "<< std::endl;
    //                     //bulk load进一个alexdatanode里 然后直接Break就可以
    //                     //所有指向这个datanode的指针也都得更新
    //                     // V* value_pairs = new V[size];
    //                     std::vector<std::pair<T, P>> value_pairs;
    //                     // 填充数组，将 keys 和 values 转换为 std::pair<T, P>
    //                     for (int i = 0; i < size; ++i) {
    //                         // value_pairs[i] = std::make_pair(keys[i], values[i]);
    //                         value_pairs.emplace_back(keys[i], values[i]);
    //                         //一会把位置0的键输出出来 在这判断一下 好像只有这里可能会改变alexdatanode的地址
    //                     }
    //                     auto data_node = new (data_node_allocator().allocate(1))
    //                         alex::AlexDataNode<T, P>(1, derived_params_.max_data_node_slots,
    //                                         key_less_, allocator_);
    //                     data_node->bulk_load(value_pairs.data(), size);
    //                     // node->items[pos].comp.leaf_node = data_node;
    //                     for (const auto& [key, value] : value_pairs) {
    //                         auto [node, item_i] = build_at(root, key);
    //                         node->items[item_i].comp.leaf_node = data_node;  // 为每个 key 设置 leaf_node
    //                                     // if(key == 224042607){
    //                                     //     std::cout << " node " << node << " item_i " << item_i << std::endl;
    //                                     // } 
    //                     }
    //                                         // if(key == 224042607){
    //                                         //     std::cout << " aaxxxalex " << alexnode << "  " << alexalex << " node " << node << " pos " << pos << std::endl;
    //                                         //     std::cout << " node->items[pos].comp.leaf_node " << node->items[pos].comp.leaf_node << std::endl;
    //                                         // } 

    //                     // delete alexnode;
    //                     // alexnode = nullptr;
    //                     delete_alexdatanode(alexnode);
    //                 } else {
    //                     // std::cout << " 333 "<< std::endl;
    //                     // BITMAP_SET(node->child_bitmap, pos); //pos和node每次循环都得重新算，直接split down一次保证加进去，不需要循环
    //                     int bug = 0;
    //                             // if(key == 218852026){
    //                             //     bug = 1;
    //                             // }
    //                     // Node* child_node = insert_build_fmcd(keys, values, size, bug);
    //                     // Node* child_node = new_nodes(1);
    //                                 // auto alexnode = node->items[pos].comp.leaf_node;
    //                                 // if(key == 224042607){
    //                                 //     std::cout << " before target_child_node->items[0].comp.leaf_node " << target_child_node->items[0].comp.leaf_node << std::endl;
    //                                 //     // temp = 1;
    //                                 // }
    //                    Node* child_node = insert_build_fmcd(keys, values, size, bug);
    //                                 // if(key == 224042607){
    //                                 //     std::cout << " after target_child_node->items[0].comp.leaf_node " << target_child_node->items[0].comp.leaf_node << std::endl;
    //                                 //     // temp = 1;
    //                                 // }
    //                                         // if(key == 224042607){
    //                                         //     std::cout << "  看每一次的一不一样child_node: " << child_node << std::endl;
    //                                         // }
    //                                         // // std::cout << "  看每一次的一不一样child_node: " << child_node << std::endl; 每一次的地址都不一样
    //                                         if(std::abs(child_node->model.a - 0.741327) < 1e-6){
    //                                             xxx = 1;
    //                                             std::cout << "node->model.a: " << node->model.a << "  node->model.b: " << node->model.b << "  node: " << node << " key " << key  << " pos " << pos << std::endl;
    //                                             std::cout << "child_node->model.a: " << child_node->model.a << "  child_node->model.b: " << child_node->model.b << "  child_node: " << child_node << std::endl;

    //                                                 std::cout << "2222BITMAP_GET(child_node->none_bitmap, 0) " << BITMAP_GET(child_node->none_bitmap, 0)<< std::endl;
    //                                                 std::cout << "2222BITMAP_GET(child_node->child_bitmap, 0) " << BITMAP_GET(child_node->child_bitmap, 0)<< std::endl;
    //                                                 std::cout << "2222child_node " << child_node << "  child_node->items[0].comp.leaf_node " << child_node->items[0].comp.leaf_node << std::endl; 
    //                                                 std::cout << "alexnode " << alexnode << std::endl;                  
    //                                             target_child_node = child_node;
    //                                             test_node = child_node;  
    //                                             alexalex =  child_node->items[0].comp.leaf_node;
    //                                         }
    //                     // std::cout << " 33333333333 "<< std::endl;
    //                     // std::unordered_map<Node*, std::vector<int>> node_item_map;
    //                     // for (int i = 0; i < size; ++i) {
    //                     //     const auto& key = keys[i];
    //                     //     // std::cout << " key " << key << " size " << size  << " i " << i << std::endl;
    //                     //     auto [node1, item_i] = build_at(root, key);
    //                     //     // if(item_i == pos){
    //                     //     //     std::cout << " node " << node << "  node1 " << node1 << "  item_i " << item_i << " pos " << pos << std::endl;
    //                     //     // }
    //                     //     // if (node1 == nullptr) {
    //                     //     //     std::cerr << "Error: build_at returned a nullptr for key " << key << std::endl;
    //                     //     //     exit(1);
    //                     //     // }
    //                     //     node_item_map[node1].push_back(item_i);
    //                     //     // BITMAP_CLEAR(node->none_bitmap, item_i);
    //                     //     // BITMAP_SET(node->child_bitmap, item_i);
    //                     //     // node->items[item_i].comp.child = child_node;
    //                     // }
    //                     // for (const auto& [node2, item_indices] : node_item_map) {
    //                     //     for (auto item_i : item_indices) {
    //                     //         BITMAP_SET(node2->child_bitmap, item_i);
    //                     //         // if (child_node == nullptr) {
    //                     //         //     std::cerr << "Error: child_node is nullptr" << std::endl;
    //                     //         //     exit(1);
    //                     //         // }
    //                     //         node2->items[item_i].comp.child = child_node;
    //                     //     }
    //                     // }
    //                     std::set<std::pair<Node*, int>> node_item_set;

    //                     for (int i = 0; i < size; ++i) {
    //                         const auto& key = keys[i];
    //                         auto [node1, item_i] = build_at(root, key);

    //                         if (node1 == nullptr) {
    //                             std::cerr << "Error: build_at returned a nullptr for key " << key << std::endl;
    //                             exit(1);
    //                         }
    //                         node1->items[item_i].comp.leaf_node = nullptr;
    //                         // 插入 (node1, item_i) 组合，std::set 自动保证唯一性
    //                         node_item_set.emplace(node1, item_i);
    //                     }

    //                     // 遍历不重复的 (Node*, int) 组合
    //                     for (const auto& [node2, item_i] : node_item_set) {
    //                         BITMAP_SET(node2->child_bitmap, item_i);

    //                         if (child_node == nullptr) {
    //                             std::cerr << "Error: child_node is nullptr" << std::endl;
    //                             exit(1);
    //                         }

    //                         node2->items[item_i].comp.child = child_node;
    //         // if(key == 218852026){
    //         //     std::cout << "cccnode2 " << node2 << " item_i " << item_i << " child_node " << child_node << std::endl;
    //         // }
    //                     }
    //                     // delete alexnode;
    //                     // alexnode = nullptr;
    //                     delete_alexdatanode(alexnode);
    //                     // child_node = nullptr;
    //                     // std::cout << " rrrrrrrchild_node " << child_node << std::endl;
    //                     // delete_nodes(child_node,1);
    //                     // std::cout << " 444444444444 "<< std::endl;
    //                     // auto first_pair = *node_item_set.begin();
    //                     // Node* node_ptr = first_pair.first;
    //                     // int item_index = first_pair.second;
    //                                     // if(std::abs(child_node->model.a - 0.741327) < 1e-6){
    //                                     //         std::cout << "333child_node " << child_node << "  child_node->items[0].comp.leaf_node " << child_node->items[0].comp.leaf_node << std::endl;                    
    //                                     //         auto nodenode = node_ptr->items[item_index].comp.child;
    //                                     //         std::cout << "333nodenode " << nodenode << "  nodenode->items[0].comp.leaf_node " << nodenode->items[0].comp.leaf_node << std::endl;

    //                                     // } 
    //                     // node->items[pos].comp.child = child_node;
    //                 }
                    
    //                 // 释放临时数组
    //                 delete[] keys;
    //                 delete[] values;
    //                 // std::cout << " 444 "<< std::endl;
    //                 //或者就是从中间把alexdatanode分开 但是这样是不是不能保证上层完全精确啊 还是得根据上层来区分下层的叶子节点

    //                 // //split down之后得重新找下叶子节点！！！！！！！！！！！！！！！！下面代码不对，但是是这个意思
    //                 // //如果是把所有的键拿出来然后重新fmcd的话就已经把键加进去了 就不需要了，因为已经加进去了
    //                 // leaf = static_cast<data_node_type*>(parent->get_child_node(key));
    //                 break;
    //             }
    //             // insert_to_data = 1;
    //             break;
    //         } else {
    //             node = node->items[pos].comp.child;
    //         }
    //     }
    //     // for (int i = 0; i < path_size; i ++) {
    //     //     path[i]->num_insert_to_data += insert_to_data;
    //     // }

    //     // for (int i = 0; i < path_size; i ++) {
    //     //     Node* node = path[i];
    //     //     const int num_inserts = node->num_inserts;
    //     //     const int num_insert_to_data = node->num_insert_to_data;
    //     //     const bool need_rebuild = node->fixed == 0 && node->size >= node->build_size * 2 && node->size >= 64 && num_insert_to_data * 10 >= num_inserts;

    //     //     if (need_rebuild) {
    //     //         const int ESIZE = node->size;
    //     //         T* keys = new T[ESIZE];
    //     //         P* values = new P[ESIZE];

    //     //         #if COLLECT_TIME
    //     //         auto start_time_scan = std::chrono::high_resolution_clock::now();
    //     //         #endif
    //     //         scan_and_destory_tree(node, keys, values);
    //     //         #if COLLECT_TIME
    //     //         auto end_time_scan = std::chrono::high_resolution_clock::now();
    //     //         auto duration_scan = end_time_scan - start_time_scan;
    //     //         stats.time_scan_and_destory_tree += std::chrono::duration_cast<std::chrono::nanoseconds>(duration_scan).count() * 1e-9;
    //     //         #endif

    //     //         #if COLLECT_TIME
    //     //         auto start_time_build = std::chrono::high_resolution_clock::now();
    //     //         #endif
    //     //         Node* new_node = build_tree_bulk(keys, values, ESIZE);
    //     //         #if COLLECT_TIME
    //     //         auto end_time_build = std::chrono::high_resolution_clock::now();
    //     //         auto duration_build = end_time_build - start_time_build;
    //     //         stats.time_build_tree_bulk += std::chrono::duration_cast<std::chrono::nanoseconds>(duration_build).count() * 1e-9;
    //     //         #endif

    //     //         delete[] keys;
    //     //         delete[] values;

    //     //         path[i] = new_node;
    //     //         if (i > 0) {
    //     //             int pos = PREDICT_POS(path[i-1], key);
    //     //             path[i-1]->items[pos].comp.child = new_node;
    //     //         }

    //     //         break;
    //     //     }
    //     // }

    //     // if(xxx == 1 && yyy<=2){
    //     //     Node * aaa = target_child_node;
    //     //     std::cout << "target_child_node " << target_child_node << std::endl;
    //     //     std::cout << "aaa " << aaa << std::endl;
    //     //     std::cout  << "  aaa->items[0].comp.leaf_node " << aaa->items[0].comp.leaf_node << std::endl;
    //     // }


    //     if(ok) {
    //         *ok = true;
    //     }
    //     // std::cout << " 555 "<< std::endl;
    //     return path[0];
    // }

    // // SATISFY_LOWER = true means all the keys in the subtree of `node` are no less than to `lower`.
    // template<bool SATISFY_LOWER>
    // int range_core_len(std::pair <T, P> *results, int pos, Node *node, const T &lower, int len) {
    //     if constexpr(SATISFY_LOWER)
    //     {
    //         int bit_pos = 0;
    //         const bitmap_t *none_bitmap = node->none_bitmap;
    //         while (bit_pos < node->num_items) {
    //             bitmap_t not_none = ~(*none_bitmap);
    //             while (not_none) {
    //                 int latest_pos = BITMAP_NEXT_1(not_none);
    //                 not_none ^= 1 << latest_pos;

    //                 int i = bit_pos + latest_pos;
    //                 if (BITMAP_GET(node->child_bitmap, i) == 0) {
    //                     results[pos] = {node->items[i].comp.data.key, node->items[i].comp.data.value};
    //                     // __builtin_prefetch((void*)&(node->items[i].comp.data.key) + 64);
    //                     pos++;
    //                 } else {
    //                     pos = range_core_len<true>(results, pos, node->items[i].comp.child, lower, len);
    //                 }
    //                 if (pos >= len) {
    //                     return pos;
    //                 }
    //             }

    //             bit_pos += BITMAP_WIDTH;
    //             none_bitmap++;
    //         }
    //         return pos;
    //     } else {
    //         int lower_pos = PREDICT_POS(node, lower);
    //         if (BITMAP_GET(node->none_bitmap, lower_pos) == 0) {
    //             if (BITMAP_GET(node->child_bitmap, lower_pos) == 0) {
    //                 if (node->items[lower_pos].comp.data.key >= lower) {
    //                     results[pos] = {node->items[lower_pos].comp.data.key, node->items[lower_pos].comp.data.value};
    //                     pos++;
    //                 }
    //             } else {
    //                 pos = range_core_len<false>(results, pos, node->items[lower_pos].comp.child, lower, len);
    //             }
    //             if (pos >= len) {
    //                 return pos;
    //             }
    //         }
    //         if (lower_pos + 1 >= node->num_items) {
    //             return pos;
    //         }
    //         int bit_pos = (lower_pos + 1) / BITMAP_WIDTH * BITMAP_WIDTH;
    //         const bitmap_t *none_bitmap = node->none_bitmap + bit_pos / BITMAP_WIDTH;
    //         while (bit_pos < node->num_items) {
    //             bitmap_t not_none = ~(*none_bitmap);
    //             while (not_none) {
    //                 int latest_pos = BITMAP_NEXT_1(not_none);
    //                 not_none ^= 1 << latest_pos;

    //                 int i = bit_pos + latest_pos;
    //                 if (i <= lower_pos) continue;
    //                 if (BITMAP_GET(node->child_bitmap, i) == 0) {
    //                     results[pos] = {node->items[i].comp.data.key, node->items[i].comp.data.value};
    //                     // __builtin_prefetch((void*)&(node->items[i].comp.data.key) + 64);
    //                     pos++;
    //                 } else {
    //                     pos = range_core_len<true>(results, pos, node->items[i].comp.child, lower, len);
    //                 }
    //                 if (pos >= len) {
    //                     return pos;
    //                 }
    //             }
    //             bit_pos += BITMAP_WIDTH;
    //             none_bitmap++;
    //         }
    //         return pos;
    //     }
    // }
};

#endif // __LIAL_H__

void print_average_times() {
  if (search_count > 0) {
      std::cout << "LIAL Average time to find alex_node: " 
                << (alex_node_time_sum / search_count) << " seconds." << std::endl;
      std::cout << "LIAL Average time to find key in alex_node: " 
                << (find_key_time_sum / search_count) << " seconds." << std::endl;
      std::cout << "search_count: " << search_count << std::endl;
  } else {
      std::cout << "LIAL No searches performed yet." << std::endl;
  }
}

}
