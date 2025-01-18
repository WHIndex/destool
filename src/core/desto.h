#ifndef __DESTO_H__
#define __DESTO_H__

#include "desto_base.h"
#include "node.h"
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

namespace desto {

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

    typedef LIPP<T, P> self_type;

    class Iterator;

    /* User-changeable parameters */
    struct Params {
        double expected_insert_frac = 1;
        int max_node_size = 1 << 24;
        bool approximate_model_computation = true;
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

    bool insert(const T& key, const P& value) {
        bool ok = true;
        root = insert_tree(root, key, value, &ok);
        return ok;
    }

    std::pair<Node*, int> build_at(Node* build_root, const T& key) const {
        // std::cout << "build_root  " << build_root << "key" << key << std::endl;
        Node* node = build_root;
        // std::cout << "node  " << node << "node-<model.a" << node->model.a << "node->model.b" << node->model.b << std::endl;
        while (true) {
            int pos = PREDICT_POS(node, key);
            // std::cout << "pos  " << pos << "BITMAP_GET(node->child_bitmap, pos)" << BITMAP_GET(node->child_bitmap, pos) << std::endl;
            if (BITMAP_GET(node->child_bitmap, pos) == 1) {
                node = node->items[pos].comp.child;
            } else {
                return {node, pos};
            }
        }
    }

    P at(const T& key, bool skip_existence_check, bool& exist) const {
        Node* node = root;
        exist = true;
        while (true) {
            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->child_bitmap, pos) == 1) {
                node = node->items[pos].comp.child;
            } else {
                if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                    exist = false;
                    return static_cast<P>(0);
                } else{
                    auto lnode = node->items[pos].comp.leaf_node;
                    int idx = lnode->find_key(key);
                    if (idx < 0) {
                        exist = false;
                        return static_cast<P>(0);
                    } else {
                        return lnode->get_payload(idx);
                    }
                }
            }
        }
    }

    typename self_type::Iterator lower_bound(const T& key) {
        Node* node = root;
        while (true) {
            int pos = PREDICT_POS(node, key);
            if (BITMAP_GET(node->child_bitmap, pos) == 1) {
                node = node->items[pos].comp.child;
            } else {
                if (BITMAP_GET(node->none_bitmap, pos) == 1) {
                    std::cout << "error  "<< std::endl;
                    exit(1);
                } else{
                    auto leaf = node->items[pos].comp.leaf_node;
                    int idx = leaf->find_lower(key);
                    return Iterator(leaf, idx);
                }
            }
        }    
    }
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
        root = build_tree_bottom_up(keys, values, num_keys);

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
                return true;
            } else {
                node = node->items[pos].comp.child;
            }
        }
    }

    size_t total_size() const {
        std::stack < Node * > s;
        s.push(root);

        size_t size = 0;
        // size_t leaf_size = 0;
        size_t item_size = 0;
        std::unordered_set<lnode::LDataNode<T, P>*> calculated_leaf_nodes;  // 记录已经计算过的 leaf_node
        std::unordered_set<Node*> calculated_inner_nodes;
        while (!s.empty()) {
            Node *node = s.top();
            s.pop();
            size += sizeof(*node);
            size += sizeof(*(node->none_bitmap));
            size += sizeof(*(node->child_bitmap));
            for (int i = 0; i < node->num_items; i++) {
                size += sizeof(Item);
                item_size += sizeof(Item);
                if (BITMAP_GET(node->child_bitmap, i) == 1) {
                    auto inner_node = node->items[i].comp.child;
                    if (calculated_inner_nodes.find(inner_node) == calculated_inner_nodes.end()) {
                        s.push(inner_node);
                        calculated_inner_nodes.insert(inner_node);  // 记录该 leaf_node 为已计算
                    }   
                } else if(BITMAP_GET(node->none_bitmap, i) == 0) {
                    auto leaf_node = node->items[i].comp.leaf_node;
                    if (calculated_leaf_nodes.find(leaf_node) == calculated_leaf_nodes.end()) {
                        size += leaf_node->data_size();
                        // leaf_size += leaf_node->data_size();
                        calculated_leaf_nodes.insert(leaf_node);  // 记录该 leaf_node 为已计算
                    }
                }
            }
        }
        return size;
    }

private:
    struct Node;
    struct Item
    {
        union {
            Node* child;
            lnode::LDataNode<T,P>* leaf_node;
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
    
    using Compare = lnode::NodeCompare;
    using Alloc = std::allocator<std::pair<T, P>>;
    Compare key_less_ = Compare();
    Alloc allocator_ = Alloc();

    typename lnode::LDataNode<T,P>::alloc_type data_node_allocator() {
        return typename lnode::LDataNode<T,P>::alloc_type(allocator_);
    }

    void delete_ldatanode(lnode::LDataNode<T,P>* node) {
        if (node == nullptr) {
        return;
        } else if (node->is_leaf_) {
        data_node_allocator().destroy(static_cast<lnode::LDataNode<T,P>*>(node));
        data_node_allocator().deallocate(static_cast<lnode::LDataNode<T,P>*>(node), 1);
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

            const int begin = begin2;
            const int end = end2;
            const int level = 0;
            Node* node = nodenode;

                T* keys = _keys + begin;
                P* values = _values + begin;
                const int size = end - begin;
                const int BUILD_GAP_CNT = compute_gap_count(size);

                const int max_items = 20000000; // 尝试将num_items控制的最大阈值为 20000000

                node->is_two = 0;
                node->build_size = size;
                node->size = size;
                node->fixed = 0;
                node->num_inserts = node->num_insert_to_data = 0;

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
                if (node->num_items > max_items) {
                    double scale_factor = static_cast<double>(max_items) / node->num_items;
                    node->model.a *= scale_factor;
                    node->model.b *= scale_factor;
                    node->num_items = max_items;
                }
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
                        // BITMAP_SET(node->child_bitmap, item_i);
                        node->items[item_i].comp.child = new_nodes(1);
                        V* value_pairs = new V[1];
                        value_pairs[0] = std::make_pair(keys[offset], values[offset]);
                        auto data_node = new (data_node_allocator().allocate(1))
                            lnode::LDataNode<T, P>(level + 1, derived_params_.max_data_node_slots,
                                            key_less_, allocator_);    
                        lnode::LinearModel<T> data_node_model;
                        lnode::LDataNode<T,P>::build_model(value_pairs, 1, &data_node_model, params_.approximate_model_computation);
                        lnode::DataNodeStats stats;
                        data_node->cost_ = lnode::LDataNode<T,P>::compute_expected_cost(
                            value_pairs, 1, lnode::LDataNode<T,P>::kInitDensity_,
                            params_.expected_insert_frac, &data_node_model,
                            params_.approximate_cost_computation, &stats);

                        cost += data_node->cost_; 
                    } else {
                        BITMAP_CLEAR(node->none_bitmap, item_i);
                        const int num_keys = next - offset;
                        V* value_pairs = new V[num_keys];

                        // 填充数组，将 keys 和 values 转换为 std::pair<T, P>
                        for (int i = 0; i < num_keys; ++i) {
                            value_pairs[i] = std::make_pair(_keys[begin + offset + i], _values[begin + offset + i]);
                        }

                        auto data_node = new (data_node_allocator().allocate(1))
                            lnode::LDataNode<T, P>(level + 1, derived_params_.max_data_node_slots,
                                            key_less_, allocator_);

                        lnode::LinearModel<T> data_node_model;
                        lnode::LDataNode<T,P>::build_model(value_pairs, num_keys, &data_node_model, params_.approximate_model_computation);
                        lnode::DataNodeStats stats;
                        data_node->cost_ = lnode::LDataNode<T,P>::compute_expected_cost(
                            value_pairs, num_keys, lnode::LDataNode<T,P>::kInitDensity_,
                            params_.expected_insert_frac, &data_node_model,
                            params_.approximate_cost_computation, &stats);
                    }
                    if (next >= size) {
                        break;
                    } else {
                        item_i = next_i;
                        offset = next;
                    }
                }
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
    }

    /// bulk build, _keys must be sorted in asc order.
    /// FMCD method.
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
        s.push((Segment){0, _size, 1, ret});

        int level_num = 0;
        int max_level = 0;
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
                // const int BUILD_GAP_CNT = 5; //insert

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
        return ret;
    }

    Node* insert_build_fmcd(T* _keys, P* _values, int _size)
    {        
        RT_ASSERT(_size > 1);
        Node* ret = new_nodes(1);
        Node* retret = new_nodes(1);


            const int begin = 0;
            const int end = _size;
            const int level = 1;
            Node* node = retret;

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
                    BITMAP_CLEAR(node->none_bitmap, item_i);
                    const int num_keys = next - offset;
                    V* value_pairs = new V[num_keys];
                    for (int i = 0; i < num_keys; ++i) {
                        value_pairs[i] = std::make_pair(_keys[begin + offset + i], _values[begin + offset + i]);
                    }

                    auto data_node = new (data_node_allocator().allocate(1))
                        lnode::LDataNode<T, P>(level + 1, derived_params_.max_data_node_slots,
                                        key_less_, allocator_);
                    data_node->bulk_load(value_pairs, num_keys);
                    node->items[item_i].comp.leaf_node = data_node;
                    if (next >= size) {
                        break;
                    } else {
                        item_i = next_i;
                        offset = next;
                    }
            }
        return retret;
    }

    Node* build_tree_bottom_up(T* _keys, P* _values, int _size)
    {
        // RT_ASSERT(_size > 1);
        std::vector<std::pair<T, P>> fk_values;
        std::vector<std::pair<T, P>> key_value;
        key_value.reserve(_size);  // 预分配空间以提高性能
        for (int i = 0; i < _size; ++i) {
            key_value.emplace_back(_keys[i], _values[i]);
        }
        // first_keys = desto::internal::segment_linear_optimal_model_fk(key_value, _size, 64);
        fk_values = desto::internal::segment_linear_optimal_model_fk_value(key_value, _size, 64);
        int fk_size = fk_values.size();
        std::vector<T> first_keys(fk_size);
        for (size_t i = 0; i < fk_size; ++i) {
            first_keys[i] = fk_values[i].first; // 提取每个段的第一个键
        }

        Node * build_root;
        if (fk_size == 1) {
            build_root = build_tree_none();
        } else {
            build_root = build_tree_bulk_fmcd(first_keys.data(), _values, first_keys.size());
        }
        int segment_count = first_keys.size();

        std::vector<bool> modified_flags(segment_count + 1, false);
        
        for (int i = 2; i < segment_count+1; ++i) {
            int start_idx = std::distance(_keys, std::lower_bound(_keys, _keys + _size, first_keys[i - 1]));
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
        int current_index = 0;
        lnode::LDataNode<T, P>* prev_leaf = nullptr;
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
                lnode::LDataNode<T, P>(1, derived_params_.max_data_node_slots, key_less_, allocator_);
            if (num_keys > 0) {                
                // 执行批量加载
                data_node->bulk_load(value_pairs.data(), num_keys);
            }

            if (prev_leaf != nullptr) {
                prev_leaf->next_leaf_ = data_node;  // 将前一个节点的 next 指向当前节点
                data_node->prev_leaf_ = prev_leaf;  // 当前节点的 prev 指向前一个节点
            }

            // 遍历 data_node 里的每个 key 并设置 item_i
            for (const auto& [key, value] : value_pairs) {
                auto [node, item_i] = build_at(build_root, key);
                BITMAP_CLEAR(node->none_bitmap, item_i);
                node->items[item_i].comp.leaf_node = data_node;  // 为每个 key 设置 leaf_node
            }

            prev_leaf = data_node;

        }
        return build_root;
    }
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
                BITMAP_CLEAR(node->none_bitmap, pos);
                std::pair<T, P> value_pair{key, value};
                auto data_node = new (data_node_allocator().allocate(1))
                    lnode::LDataNode<T, P>(1, derived_params_.max_data_node_slots, key_less_, allocator_);
                data_node->bulk_load(&value_pair, 1);
                node->items[pos].comp.leaf_node = data_node;
                // BITMAP_CLEAR(node->child_bitmap, pos); 如果出问题了 可以试试打开这句话 Bitmap会被datanode bulk_load影响到
                break;
            } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
                auto lnode = node->items[pos].comp.leaf_node;
                std::pair<int, int> ret = lnode->insert(key, value);
                int fail = ret.first;
                int insert_pos = ret.second;
                if (fail == -1) {
                    // Duplicate found and duplicates not allowed
                    *ok = false;
                    break;
                }

                if (fail) {
                    //如果第一次失败了，直接尝试一下扩大斜率，重新训练模型，如果还是不行，直接split
                    lnode->resize(lnode::LDataNode<T, P>::kMinDensity_, true,
                                lnode->is_append_mostly_right(),
                                lnode->is_append_mostly_left());
                    lnode->expected_avg_exp_search_iterations_ = 0;
                    lnode->expected_avg_shifts_ = 0;
                    lnode->reset_stats();
                    // Try again to insert the key
                    ret = lnode->insert(key, value);
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
                    int size = lnode->num_keys_ + 1;
                    T* keys = new T[size];
                    P* values = new P[size];
                    lnode->extract_keys_and_values_with_insertion(keys, values, size, key, value);
                    if (size <= 2) {
                        std::vector<std::pair<T, P>> value_pairs;
                        for (int i = 0; i < size; ++i) {
                            value_pairs.emplace_back(keys[i], values[i]);
                        }
                        auto data_node = new (data_node_allocator().allocate(1))
                            lnode::LDataNode<T, P>(1, derived_params_.max_data_node_slots,
                                            key_less_, allocator_);
                        data_node->bulk_load(value_pairs.data(), size);
                        for (const auto& [key, value] : value_pairs) {
                            auto [node, item_i] = build_at(root, key);
                            node->items[item_i].comp.leaf_node = data_node;  // 为每个 key 设置 leaf_node
                        }
                        delete_ldatanode(lnode);
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
                        delete_ldatanode(lnode);
                    }  
                    // 释放临时数组
                    delete[] keys;
                    delete[] values;
                    break;
                }
                break;
            } else {
                node = node->items[pos].comp.child;
            }
        }
        if(ok) {
            *ok = true;
        }
        return path[0];
    }

    //Iterator
 public:
  class Iterator {
   public:
    lnode::LDataNode<T, P>* cur_leaf_ = nullptr;  // current data node
    int cur_idx_ = 0;         // current position in key/data_slots of data node
    int cur_bitmap_idx_ = 0;  // current position in bitmap
    uint64_t cur_bitmap_data_ = 0;  // caches the relevant data in the current
                                    // bitmap position

    Iterator() {}

    Iterator(lnode::LDataNode<T, P>* leaf, int idx) : cur_leaf_(leaf), cur_idx_(idx) {
      initialize();
    }

    Iterator(const Iterator& other)
        : cur_leaf_(other.cur_leaf_),
          cur_idx_(other.cur_idx_),
          cur_bitmap_idx_(other.cur_bitmap_idx_),
          cur_bitmap_data_(other.cur_bitmap_data_) {}

    // Iterator(const ReverseIterator& other)
    //     : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
    //   initialize();
    // }

    Iterator& operator=(const Iterator& other) {
      if (this != &other) {
        cur_idx_ = other.cur_idx_;
        cur_leaf_ = other.cur_leaf_;
        cur_bitmap_idx_ = other.cur_bitmap_idx_;
        cur_bitmap_data_ = other.cur_bitmap_data_;
      }
      return *this;
    }

    Iterator& operator++() {
      advance();
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      advance();
      return tmp;
    }

    V operator*() const {
      return std::make_pair(cur_leaf_->key_slots_[cur_idx_],
                            cur_leaf_->payload_slots_[cur_idx_]);
    }


    const T& key() const { return cur_leaf_->get_key(cur_idx_); }

    P& payload() const { return cur_leaf_->get_payload(cur_idx_); }

    bool is_end() const { return cur_leaf_ == nullptr; }

    bool operator==(const Iterator& rhs) const {
      return cur_idx_ == rhs.cur_idx_ && cur_leaf_ == rhs.cur_leaf_;
    }

    bool operator!=(const Iterator& rhs) const { return !(*this == rhs); };

   private:
    void initialize() {
      if (!cur_leaf_) return;
      assert(cur_idx_ >= 0);
      if (cur_idx_ >= cur_leaf_->data_capacity_) {
        cur_leaf_ = cur_leaf_->next_leaf_;
        cur_idx_ = 0;
        if (!cur_leaf_) return;
      }

      cur_bitmap_idx_ = cur_idx_ >> 6;
      cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];

      // Zero out extra bits
      int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
      cur_bitmap_data_ &= ~((1ULL << bit_pos) - 1);

      (*this)++;
    }

    forceinline void advance() {
      while (cur_bitmap_data_ == 0) {
        cur_bitmap_idx_++;
        if (cur_bitmap_idx_ >= cur_leaf_->bitmap_size_) {
          cur_leaf_ = cur_leaf_->next_leaf_;
          cur_idx_ = 0;
          if (cur_leaf_ == nullptr) {
            return;
          }
          cur_bitmap_idx_ = 0;
        }
        cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
      }
      uint64_t bit = desto::lnode::extract_rightmost_one(cur_bitmap_data_);
      cur_idx_ = desto::lnode::get_offset(cur_bitmap_idx_, bit);
      cur_bitmap_data_ = desto::lnode::remove_rightmost_one(cur_bitmap_data_);
    }
  };    

};

#endif // __DESTO_H__
}
