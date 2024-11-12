// This file is part of PGM-index <https://github.com/gvinciguerra/PGM-index>.
// Copyright (c) 2018 Giorgio Vinciguerra.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
// #include "alex_nodes.h"
// #include "lippwhh.h"
// #include "segmentInterface.h"

#ifdef _OPENMP
#include <omp.h>
#else
#pragma message ("Compilation with -fopenmp is optional but recommended")
#define omp_get_num_procs() 1
#define omp_get_max_threads() 1
#endif

namespace lial::internal {

template<typename T>
using LargeSigned = typename std::conditional_t<std::is_floating_point_v<T>,
                                                long double,
                                                std::conditional_t<(sizeof(T) < 8), int64_t, __int128>>;

template<typename X, typename Y>
class OptimalPiecewiseLinearModel {
private:
    using SX = LargeSigned<X>;
    using SY = LargeSigned<Y>;

    struct Slope {
        SX dx{};
        SY dy{};

        bool operator<(const Slope &p) const { return dy * p.dx < dx * p.dy; }
        bool operator>(const Slope &p) const { return dy * p.dx > dx * p.dy; }
        bool operator==(const Slope &p) const { return dy * p.dx == dx * p.dy; }
        bool operator!=(const Slope &p) const { return dy * p.dx != dx * p.dy; }
        explicit operator long double() const { return dy / (long double) dx; }
    };

    struct Point {
        X x{};
        Y y{};

        Slope operator-(const Point &p) const { return {SX(x) - p.x, SY(y) - p.y}; }
    };

    const Y epsilon;
    std::vector<Point> lower;
    std::vector<Point> upper;
    X first_x = 0;
    X last_x = 0;
    size_t lower_start = 0;
    size_t upper_start = 0;
    size_t points_in_hull = 0;
    Point rectangle[4];

    auto cross(const Point &O, const Point &A, const Point &B) const {
        auto OA = A - O;
        auto OB = B - O;
        return OA.dx * OB.dy - OA.dy * OB.dx;
    }

public:

    class CanonicalSegment;

    explicit OptimalPiecewiseLinearModel(Y epsilon) : epsilon(epsilon), lower(), upper() {
        if (epsilon < 0)
            throw std::invalid_argument("epsilon cannot be negative");

        upper.reserve(1u << 16);
        lower.reserve(1u << 16);
    }

    bool add_point(const X &x, const Y &y) {
        if (points_in_hull > 0 && x <= last_x){
            // std::cout << "points_in_hull " << points_in_hull << "  x  "  << x << "   last_x   " << last_x << std::endl;
            throw std::logic_error("Points must be increasing by x.");
        }


        last_x = x;
        auto max_y = std::numeric_limits<Y>::max();
        auto min_y = std::numeric_limits<Y>::lowest();
        Point p1{x, y >= max_y - epsilon ? max_y : y + epsilon};
        Point p2{x, y <= min_y + epsilon ? min_y : y - epsilon};

        if (points_in_hull == 0) {
            first_x = x;
            rectangle[0] = p1;
            rectangle[1] = p2;
            upper.clear();
            lower.clear();
            upper.push_back(p1);
            lower.push_back(p2);
            upper_start = lower_start = 0;
            ++points_in_hull;
            return true;
        }

        if (points_in_hull == 1) {
            rectangle[2] = p2;
            rectangle[3] = p1;
            upper.push_back(p1);
            lower.push_back(p2);
            ++points_in_hull;
            return true;
        }

        auto slope1 = rectangle[2] - rectangle[0];
        auto slope2 = rectangle[3] - rectangle[1];
        bool outside_line1 = p1 - rectangle[2] < slope1;
        bool outside_line2 = p2 - rectangle[3] > slope2;

        if (outside_line1 || outside_line2) {
            points_in_hull = 0;
            return false;
        }

        if (p1 - rectangle[1] < slope2) {
            // Find extreme slope
            auto min = lower[lower_start] - p1;
            auto min_i = lower_start;
            for (auto i = lower_start + 1; i < lower.size(); i++) {
                auto val = lower[i] - p1;
                if (val > min)
                    break;
                min = val;
                min_i = i;
            }

            rectangle[1] = lower[min_i];
            rectangle[3] = p1;
            lower_start = min_i;

            // Hull update
            auto end = upper.size();
            for (; end >= upper_start + 2 && cross(upper[end - 2], upper[end - 1], p1) <= 0; --end)
                continue;
            upper.resize(end);
            upper.push_back(p1);
        }

        if (p2 - rectangle[0] > slope1) {
            // Find extreme slope
            auto max = upper[upper_start] - p2;
            auto max_i = upper_start;
            for (auto i = upper_start + 1; i < upper.size(); i++) {
                auto val = upper[i] - p2;
                if (val < max)
                    break;
                max = val;
                max_i = i;
            }

            rectangle[0] = upper[max_i];
            rectangle[2] = p2;
            upper_start = max_i;

            // Hull update
            auto end = lower.size();
            for (; end >= lower_start + 2 && cross(lower[end - 2], lower[end - 1], p2) >= 0; --end)
                continue;
            lower.resize(end);
            lower.push_back(p2);
        }

        ++points_in_hull;
        return true;
    }

    CanonicalSegment get_segment() {
        if (points_in_hull == 1)
            return CanonicalSegment(rectangle[0], rectangle[1], first_x);
        return CanonicalSegment(rectangle, first_x);
    }

    void reset() {
        points_in_hull = 0;
        lower.clear();
        upper.clear();
    }
};

template<typename X, typename Y>
class OptimalPiecewiseLinearModel<X, Y>::CanonicalSegment {
    friend class OptimalPiecewiseLinearModel;

    Point rectangle[4];
    X first;

    CanonicalSegment(const Point &p0, const Point &p1, X first) : rectangle{p0, p1, p0, p1}, first(first) {};

    CanonicalSegment(const Point (&rectangle)[4], X first)
        : rectangle{rectangle[0], rectangle[1], rectangle[2], rectangle[3]}, first(first) {};

    bool one_point() const {
        return rectangle[0].x == rectangle[2].x && rectangle[0].y == rectangle[2].y
            && rectangle[1].x == rectangle[3].x && rectangle[1].y == rectangle[3].y;
    }

public:

    CanonicalSegment() = default;

    X get_first_x() const { return first; }

    std::pair<long double, long double> get_intersection() const {
        auto &p0 = rectangle[0];
        auto &p1 = rectangle[1];
        auto &p2 = rectangle[2];
        auto &p3 = rectangle[3];
        auto slope1 = p2 - p0;
        auto slope2 = p3 - p1;

        if (one_point() || slope1 == slope2)
            return {p0.x, p0.y};

        auto p0p1 = p1 - p0;
        auto a = slope1.dx * slope2.dy - slope1.dy * slope2.dx;
        auto b = (p0p1.dx * slope2.dy - p0p1.dy * slope2.dx) / static_cast<long double>(a);
        auto i_x = p0.x + b * slope1.dx;
        auto i_y = p0.y + b * slope1.dy;
        return {i_x, i_y};
    }

    std::pair<long double, SY> get_floating_point_segment(const X &origin) const {
        if (one_point())
            return {0, (rectangle[0].y + rectangle[1].y) / 2};

        if constexpr (std::is_integral_v<X> && std::is_integral_v<Y>) {
            auto slope = rectangle[3] - rectangle[1];
            auto intercept_n = slope.dy * (SX(origin) - rectangle[1].x);
            auto intercept_d = slope.dx;
            auto rounding_term = ((intercept_n < 0) ^ (intercept_d < 0) ? -1 : +1) * intercept_d / 2;
            auto intercept = (intercept_n + rounding_term) / intercept_d + rectangle[1].y;
            return {static_cast<long double>(slope), intercept};
        }

        auto[i_x, i_y] = get_intersection();
        auto[min_slope, max_slope] = get_slope_range();
        auto slope = (min_slope + max_slope) / 2.;
        auto intercept = i_y - (i_x - origin) * slope;
        return {slope, intercept};
    }

    std::pair<long double, long double> get_slope_range() const {
        if (one_point())
            return {0, 1};

        auto min_slope = static_cast<long double>(rectangle[2] - rectangle[0]);
        auto max_slope = static_cast<long double>(rectangle[3] - rectangle[1]);
        return {min_slope, max_slope};
    }
};

// template<typename KEY_TYPE, typename data_type>
// struct Segment {
//     std::vector<std::pair<KEY_TYPE, data_type>> data;  // 段内所有的键值对
//     double slope;                                        // 段的斜率
//     double intercept;                                    // 段的截距
//     fiting::alex::AlexDataNode<key_type, data_type> alex_data_node;                       // 添加AlexDataNode对象

//     // 无参构造函数
//     Segment() : slope(0), intercept(0) {}  // 默认构造函数，设置默认值

//     // 构造函数，初始化斜率和截距
//     Segment(double s, double i) : slope(s), intercept(i) {}

//     // 添加键值对到段中
//     void add_data(const KEY_TYPE& key, const data_type& value) {
//         data.push_back(std::make_pair(key, value));
//     }

//     // 获取段中的键值对数量
//     size_t size() const {
//         return data.size();
//     }

//     // 调用AlexDataNode的bulk_load方法
//     void load_into_alex() {
//         std::vector<std::pair<key_type, data_type>> key_value_pairs;  // 用于存储键值对
//         for (const auto& kv : data) {
//             key_value_pairs.push_back(kv);  // 提取每个键值对
//         }

//         // 构建线性模型并加载数据
//         fiting::alex::LinearModel<key_type> model(slope, intercept);

//         // 确保传递的是键值对数组，并调用 bulk_load
//         alex_data_node.bulk_load(key_value_pairs.data(), key_value_pairs.size(), model, false);
//     }

//     // 返回 const AlexDataNode 的引用（用于 const 语境）
//     const fiting::alex::AlexDataNode<key_type, data_type>& get_alex_data_node() const {
//         return alex_data_node;
//     }

//     // 返回非 const 的 AlexDataNode 引用（用于可修改场景）
//     fiting::alex::AlexDataNode<key_type, data_type>& get_alex_data_node() {
//         return alex_data_node;
//     }

//     // 打印段的信息（辅助调试）
//     void print() const {
//         std::cout << "Segment [slope=" << slope << ", intercept=" << intercept << ", datasize=" << data.size()  << "]\n";
//         // for (const auto& kv : data) {
//         //     std::cout << "Key: " << kv.first << " | Value: " << kv.second << "\n";
//         // }
//     }
// };

// template<typename KEY_TYPE, typename data_type>
// class Segment;

// template<typename KEY_TYPE, typename data_type>
// std::vector<Segment<KEY_TYPE, data_type>> segment_linear_optimal_model(
//     std::vector<std::pair<KEY_TYPE, data_type>>& key_value, size_t num_elements, size_t epsilon);

// template<typename KEY_TYPE, typename data_type>
// class Segment {
// public:
//     using Floating = double;  // 如果需要使用 float，可以修改为 float

//     // 构造函数，增加 first_key 和 keys 参数，初始化 node 数组
//     Segment(Floating slope, int32_t intercept, KEY_TYPE first_key, std::vector<KEY_TYPE> keys)
//         : slope_(slope), intercept_(intercept), first_key_(first_key), pre_size(0) {
//         // 初始化 key 和 node 数组
//         key_count_ = keys.size();

//         ////////////////////////////////////////////////////
//         // 直接以 keys 的大小作为 key 数组的大小
//         int sizesize = keys.size()+10;
        
//         // 分配 key 数组的内存空间
//         key = new KEY_TYPE[sizesize]();
        
//         // 将 keys 数组中的元素直接拷贝到 key 数组中
//         for (size_t i = 0; i < keys.size(); ++i) {
//             key[i] = keys[i];
//         }

//         ///////////////////////////////////////////////////////////////////////////////////////////////////
//         // std::vector<std::pair<KEY_TYPE, data_type>> key_value;
//         // for (size_t i = 0; i < keys.size(); ++i) {
//         //     key_value.emplace_back(keys[i], static_cast<data_type>(i));
//         // }

//         // auto segments = fiting::internal::segment_linear_optimal_model(key_value, keys.size(), static_cast<size_t>(4));
//         // std::cout << "111111111111   segments.size() " << segments.size() << std::endl;
//         // for (auto& seg : segments) {
//         // std::cout << "11111111111   seg.get_first_key() " << seg.get_first_key() << "  seg.get_slope() " << seg.get_slope() << "  seg.get_intercept()  " << seg.get_intercept() << std::endl;
//         // std::cout << "11111111111   seg.get_first_key()  " << seg.get_first_key() << "  seg.last_key  " << seg.get_keys()[seg.get_key_count()-1] << "  seg.get_key_count() " << seg.get_key_count() << " lastkey_pre_pos " << static_cast<int>(std::round(seg.get_slope() * seg.get_keys()[seg.get_key_count()-1]) + seg.get_intercept()) << std::endl;
//         // }
//         /////////////////////////////////////////////////////////////////////////////////////////////////

//         // pre_size 就是 keys 的大小
//         pre_size = sizesize;  
//         ///////////////////////////////////////////////////////      
//                             // // int sizesize = static_cast<int>(slope * keys[keys.size()-1] + intercept) + 1000;
//                             // int sizesize = static_cast<int>(slope * keys.back() + intercept) + 1;
//                             // key = new KEY_TYPE[sizesize]();
//                             // // 计算 bitmap 的大小，以64位为一个单位
//                             // bitmap_size_ = (sizesize + 63) / 64;
//                             // bitmap_ = new uint64_t[bitmap_size_]();
//                             // // int last_predicted_position = -1;
//                             // for (size_t i = 0; i < keys.size(); ++i) {
//                             //     int predicted_position = static_cast<int>(slope * keys[i] + intercept);
//                             //     // int insert_position = binary_search_within_error(predicted_position, 4);  // 在误差4内查找合适位置
//                             //     // if (insert_position == -1) {
//                             //     //     std::cerr << "No valid position found for key: " << keys[i] << std::endl;
//                             //     //     exit(1);
//                             //     // }

//                             //     // // 存储键，并设置对应的 bitmap 位
//                             //     // int bitmap_pos = insert_position >> 6;
//                             //     // int bit_pos = insert_position & 63;
//                             //     // bitmap_[bitmap_pos] |= (1ULL << bit_pos);

//                             //     // // 更新 pre_size 为最大插入位置 + 1
//                             //     // pre_size = std::max(pre_size, insert_position + 1);

//                             //     // while (predicted_position <= last_predicted_position) {
//                             //     //     // 如果 predicted_position 小于或等于上一个位置，则顺序插入
//                             //     //     predicted_position ++;
//                             //     // }
//                             //     // // std::cout << "after predicted_position  " << predicted_position << std::endl;
//                             //     // if(predicted_position >= sizesize){
//                             //     //     std::cerr << "Predicted position out of bounds: " << predicted_position << " sizesize " << sizesize << std::endl;
//                             //     //     exit(1);
//                             //     // }
//                             //     // key[predicted_position] = keys[i];  // 复制键值
//                             //     // int bitmap_pos = predicted_position >> 6;
//                             //     // int bit_pos = predicted_position - (bitmap_pos << 6);
//                             //     // bitmap_[bitmap_pos] |= (1ULL << bit_pos);
//                             //     // last_predicted_position = predicted_position;
//                             //     // if(i == keys.size()-1){
//                             //     //     pre_size = predicted_position + 1 ;
//                             //     //     std::cout << "slope  " << slope << "  intercept  " << intercept << " keys.size()  " << keys.size() << "  pre_size  " << pre_size  << std::endl;
//                             //     //     // std::cout << "predicted_position  " << predicted_position << "  key[predicted_position]  " << key[predicted_position] << "  keys[i]  " << keys[i] << std::endl;
//                             //     // }
//                             // }
//         // std::cout << "seg first_key  " << first_key << "  key[58]  " << key[58] << std::endl;
//         // node = new SegmentInterface<KEY_TYPE, data_type>*[pre_size]();  // 初始化为空指针数组
//     }

//     Segment(const Segment& other) {
//         slope_ = other.slope_;
//         intercept_ = other.intercept_;
//         first_key_ = other.first_key_;
//         key_count_ = other.key_count_;
//         pre_size = other.pre_size;

//         // 深拷贝 key 数组
//         key = new KEY_TYPE[other.pre_size];
//         std::copy(other.key, other.key + other.pre_size, key);

//         // 深拷贝 bitmap 数组
//         bitmap_size_ = other.bitmap_size_;
//         bitmap_ = new uint64_t[bitmap_size_];
//         std::copy(other.bitmap_, other.bitmap_ + bitmap_size_, bitmap_);

//         // 深拷贝 node 指针数组
//         node = new SegmentInterface<KEY_TYPE, data_type>*[other.pre_size];
//         std::copy(other.node, other.node + other.pre_size, node);
//     }

//     Segment& operator=(const Segment& other) {
//         if (this != &other) {
//             slope_ = other.slope_;
//             intercept_ = other.intercept_;
//             first_key_ = other.first_key_;
//             key_count_ = other.key_count_;
//             pre_size = other.pre_size;

//             // 深拷贝 key 数组
//             delete[] key;  // 释放旧的 key 数组
//             key = new KEY_TYPE[other.pre_size];
//             std::copy(other.key, other.key + other.pre_size, key);

//             // 深拷贝 bitmap 数组
//             delete[] bitmap_;  // 释放旧的 bitmap 数组
//             bitmap_size_ = other.bitmap_size_;
//             bitmap_ = new uint64_t[bitmap_size_];
//             std::copy(other.bitmap_, other.bitmap_ + bitmap_size_, bitmap_);

//             // 深拷贝 node 指针数组
//             delete[] node;  // 释放旧的 node 数组
//             node = new SegmentInterface<KEY_TYPE, data_type>*[other.pre_size];
//             std::copy(other.node, other.node + other.pre_size, node);
//         }
//         return *this;
//     }


//     // 析构函数，释放 node 数组
//     ~Segment() {
//         for (int i = 0; i < pre_size; ++i) {
//             delete node[i];
//         }
//         delete[] node;  // 释放 node 数组
//         delete[] key;   // 释放 key 数组
//         delete[] bitmap_; 
//     }

//     // 获取斜率
//     Floating get_slope() const { return slope_; }

//     // 获取截距
//     int32_t get_intercept() const { return intercept_; }

//     // 获取第一个键
//     KEY_TYPE get_first_key() const { return first_key_; }

//     // 获取所有键
//     const KEY_TYPE* get_keys() const { return key; }

//     std::vector<std::pair<KEY_TYPE, data_type>> getkeyvalue(std::vector<KEY_TYPE> keys){
//         std::vector<std::pair<KEY_TYPE, data_type>> key_value;
//         for (size_t i = 0; i < keys.size(); ++i) {
//             key_value.emplace_back(keys[i], static_cast<data_type>(i));
//         }
//         return key_value;
//     }

//     // 获取段中键的数量
//     int get_key_count() const { return key_count_; }

//     int get_pre_size() const { return pre_size; }

//     // 使用 bitmap 检查 key 数组中是否有有效值
//     bool has_key(int position) const {
//         if (position < 0 || position >= pre_size) return false;

//         // 通过位操作检查 bitmap_
//         int bitmap_pos = position >> 6;
//         int bit_pos = position - (bitmap_pos << 6);
//         return static_cast<bool>(bitmap_[bitmap_pos] & (1ULL << bit_pos));
//     }

//     // 获取键值（仅在有值时）
//     KEY_TYPE get_key(int position) const {
//         if (has_key(position)) {
//             return key[position];
//         } else {
//             throw std::out_of_range("No key present at the given position");
//         }
//     }

//     int predict_position(const KEY_TYPE& key) const {
//         // 使用线性公式计算预测位置
//         // int predicted_pos = static_cast<int>(slope_ * key + intercept_);
//         int predicted_pos = static_cast<int>(std::round(slope_ * key) + intercept_);
//         // return predicted_pos;
//         return std::max(0, std::min(predicted_pos, key_count_ - 1));

//         // // 限制 predicted_pos 在 [0, keys_.size() - 1] 之间
//         // return std::max(0, std::min(predicted_pos, static_cast<int>(keys_.size() - 1)));
//     }

//     // 查找从指定位置开始，向右第一个有值的位置
//     int next_valid_position(int position) const {
//         // int pos = position;
//         // while (pos < pre_size) {
//         //     if (check_exists(pos)) {
//         //         return pos;
//         //     }
//         //     pos++;
//         // }
//         int bitmap_pos = position >> 6;
//         int bit_pos = position & 63;

//         // 首先处理当前位置所在的64位块
//         uint64_t current_bits = bitmap_[bitmap_pos] >> bit_pos;
//         if (current_bits != 0) {
//             return position + __builtin_ctzll(current_bits);  // 找到第一个为1的位
//         }

//         // 如果当前64位块之后的块有有效值，逐块查找
//         for (int i = bitmap_pos + 1; i < bitmap_size_; ++i) {
//             if (bitmap_[i] != 0) {
//                 return (i << 6) + __builtin_ctzll(bitmap_[i]);
//             }
//         }
//         return position-1;
//         // throw std::out_of_range("No valid key found in the forward direction");
//     }

//     int prev_valid_position(int position) const {
//         // int pos = position;
//         // while (pos >= 0) { //不应该是大于等于0 应该是从first_key预测的位置
//         //     if (check_exists(pos)) {
//         //         return pos;
//         //     }
//         //     pos--;
//         // }
//         int bitmap_pos = position >> 6;
//         int bit_pos = position & 63;

//         // 首先处理当前位置所在的64位块
//         uint64_t current_bits = bitmap_[bitmap_pos] & ((1ULL << bit_pos) - 1);
//         if (current_bits != 0) {
//             return (bitmap_pos << 6) + (63 - __builtin_clzll(current_bits));  // 找到第一个为1的位
//         }

//         // 如果当前64位块之前的块有有效值，逐块查找
//         for (int i = bitmap_pos - 1; i >= 0; --i) {
//             if (bitmap_[i] != 0) {
//                 return (i << 6) + (63 - __builtin_clzll(bitmap_[i]));
//             }
//         }
//         return position+1;
//         // throw std::out_of_range("No valid key found in the backward direction");
//     }


//     inline bool check_exists(int pos) const {
//         assert(pos >= 0 && pos < data_capacity_);
//         int bitmap_pos = pos >> 6;
//         int bit_pos = pos - (bitmap_pos << 6);
//         return static_cast<bool>(bitmap_[bitmap_pos] & (1ULL << bit_pos));
//     }

//     // // 预测 key 应该插入的位置，返回预测的位置
//     // int predict_position(const KEY_TYPE& key) const {
//     //     int predicted_pos = static_cast<int>(slope_ * static_cast<double>(key) + intercept_);
//     //     return std::max(0, std::min(predicted_pos, static_cast<int>(keys_.size() - 1)));  // 限制范围
//     // }

//     // // 在段中插入 node
//     // void insert_node(int position, SegmentInterface<KEY_TYPE, data_type>* new_node) {
//     //     if (position >= 0 && position < keys_.size()) {
//     //         node[position] = new_node;  // 插入 node 到指定位置
//     //     }
//     // }

//     // // 获取 node 指针数组
//     // SegmentInterface<KEY_TYPE, data_type>** get_node_array() const {
//     //     return node;
//     // }
//     uint64_t* bitmap_ = nullptr;
//     int bitmap_size_ = 0;
//     // SegmentInterface<KEY_TYPE, data_type>** node; ///< 用于存储段中各位置的指针
//     KEY_TYPE *key;

// private:
//     Floating slope_;    ///< 段的斜率
//     int32_t intercept_; ///< 段的截距
//     KEY_TYPE first_key_; ///< 段的第一个键
//     int key_count_;
//     int pre_size;  ///< 记录预测出来的位置的最后
//     // std::vector<KEY_TYPE> keys_; ///< 段中所有的键
   
// };

template<typename KEY_TYPE, typename data_type>
std::vector<KEY_TYPE> segment_linear_optimal_model_fk(
    std::vector<std::pair<KEY_TYPE, data_type>>& key_value, size_t num_elements, size_t epsilon) {
  
    // 初始化返回的段起始键向量
    std::vector<KEY_TYPE> segment_first_keys;
    
    // 初始化最优线性分段模型，使用给定的 epsilon 值
    typename lial::internal::OptimalPiecewiseLinearModel<KEY_TYPE, size_t> opt_model(static_cast<size_t>(epsilon));
    
    // 临时保存段内数据
    std::vector<std::pair<KEY_TYPE, data_type>> segment_data;

    // 遍历输入数据 key_value
    for (size_t i = 0; i < num_elements; ++i) {
        KEY_TYPE& x = key_value[i].first;  // 当前数据点的键
        const data_type& y = key_value[i].second;  // 当前数据点的值

        // 添加当前点到分段模型中，x 是键，i 是下标
        if (!opt_model.add_point(x, i)) {

            // 当前点无法加入现有段，生成一个新段
            auto segment_model = opt_model.get_segment();
            
            // 获取当前段的第一个键
            KEY_TYPE first_key = segment_data.front().first;

            // 存储该段的第一个键
            segment_first_keys.push_back(first_key);
            
            // 清空当前段的数据，并重置分段模型
            segment_data.clear();
            opt_model.reset();
            
            // 重新开始新的段
            opt_model.add_point(x, i);
            segment_data.push_back({x, y});
        } else {
            // 如果当前点可以加入段内，则继续添加到临时数据中
            segment_data.push_back({x, y});
        }
    }
    
    // 处理最后一个段
    if (!segment_data.empty()) {
        // 获取最后一段的第一个键
        KEY_TYPE first_key = segment_data.front().first;
        
        // 存储最后一个段的第一个键
        segment_first_keys.push_back(first_key);
    }

    return segment_first_keys;
}

template<typename KEY_TYPE, typename data_type>
std::vector<std::pair<KEY_TYPE, data_type>> segment_linear_optimal_model_fk_value(
    std::vector<std::pair<KEY_TYPE, data_type>>& key_value, size_t num_elements, size_t epsilon) {

    // 初始化返回的键值对向量，用来保存每段的起始键和相关数据
    std::vector<std::pair<KEY_TYPE, data_type>> segment_first_keys;

    // 初始化最优线性分段模型，使用给定的 epsilon 值
    typename lial::internal::OptimalPiecewiseLinearModel<KEY_TYPE, size_t> opt_model(static_cast<size_t>(epsilon));
    
    // 临时保存段内数据
    std::vector<std::pair<KEY_TYPE, data_type>> segment_data;

    // 遍历输入数据 key_value
    for (size_t i = 0; i < num_elements; ++i) {
        KEY_TYPE& x = key_value[i].first;  // 当前数据点的键
        const data_type& y = key_value[i].second;  // 当前数据点的值

        // 添加当前点到分段模型中，x 是键，i 是下标
        if (!opt_model.add_point(x, i)) {

            // 当前点无法加入现有段，生成一个新段
            auto segment_model = opt_model.get_segment();
            
            // 获取当前段的第一个键
            KEY_TYPE first_key = segment_data.front().first;
            data_type first_value = segment_data.front().second;  // 可以选择段内的其他数据（比如第一个数据点的值）

            // 存储该段的第一个键和相关数据
            segment_first_keys.push_back({first_key, first_value});
            
            // 清空当前段的数据，并重置分段模型
            segment_data.clear();
            opt_model.reset();
            
            // 重新开始新的段
            opt_model.add_point(x, i);
            segment_data.push_back({x, y});
        } else {
            // 如果当前点可以加入段内，则继续添加到临时数据中
            segment_data.push_back({x, y});
        }
    }

    // 处理最后一个段
    if (!segment_data.empty()) {
        // 获取最后一段的第一个键及其相关数据
        KEY_TYPE first_key = segment_data.front().first;
        data_type first_value = segment_data.front().second;

        // 存储最后一个段的第一个键和相关数据
        segment_first_keys.push_back({first_key, first_value});
    }

    return segment_first_keys;
}



// template<typename KEY_TYPE, typename data_type>
// std::vector<Segment<KEY_TYPE, data_type>> segment_linear_optimal_model(
//     std::vector<std::pair<KEY_TYPE, data_type>>& key_value, size_t num_elements, size_t epsilon) {
  
//     // 初始化返回的段向量
//     std::vector<Segment<KEY_TYPE, data_type>> underlying_segs;
    
//     // 初始化最优线性分段模型，使用给定的 epsilon 值
//     typename fiting::internal::OptimalPiecewiseLinearModel<KEY_TYPE, size_t> opt_model(static_cast<size_t>(epsilon));
    
//     // 临时保存段内数据
//     std::vector<std::pair<KEY_TYPE, data_type>> segment_data;
//     std::vector<KEY_TYPE> segment_keys;  // 用于存储当前段的所有键

//     // size_t segment_index = 0;  // 每个段内的索引，从 0 开始
    
//     // 遍历输入数据 underlying_data
//     for (size_t i = 0; i < num_elements; ++i) {
//         KEY_TYPE& x = key_value[i].first;  // 当前数据点的键
//         const data_type& y = key_value[i].second;  // 当前数据点的值

//         // 添加当前点到分段模型中，x 是键，i 是下标
//         if (!opt_model.add_point(x, i)) {

//             // 当前点无法加入现有段，生成一个新段
//             auto segment_model = opt_model.get_segment();
            
//             // 计算当前段的斜率和截距
//             auto [slope, intercept] = segment_model.get_floating_point_segment(segment_data.front().first);

//             // 获取当前段的第一个键
//             KEY_TYPE first_key = segment_data.front().first;

//             intercept = -static_cast<int>(std::round(slope * first_key));

//             // // 创建并存储 Segment 对象
//             // underlying_segs.emplace_back(static_cast<typename Segment<KEY_TYPE, data_type>::Floating>(slope), 
//             //                              static_cast<int32_t>(intercept), first_key, segment_keys);
            
//             if (segment_keys.size() > 1) {
//                 // 创建并存储 Segment 对象
//                 underlying_segs.emplace_back(static_cast<typename Segment<KEY_TYPE, data_type>::Floating>(slope), 
//                                              static_cast<int32_t>(intercept), first_key, segment_keys);
//             } else {
//                 std::cerr << "Warning: Segment too small or invalid, skipping." << std::endl;
//             }
            
//             // 清空当前段的数据和键，并重置分段模型
//             segment_data.clear();
//             segment_keys.clear();
//             opt_model.reset();
            
//             // 重新开始新的段
//             // segment_index = 0;
//             opt_model.add_point(x, i);
//             segment_data.push_back({x, y});
//             segment_keys.push_back(x);  // 记录键
//         } else {
//             // 如果当前点可以加入段内，则继续添加到临时数据和键中
//             segment_data.push_back({x, y});
//             segment_keys.push_back(x);  // 记录键
//         }
//         // ++segment_index;
//     }
    
//     // 处理最后一个段
//     if (!segment_data.empty()) {
//         auto segment_model = opt_model.get_segment();
//         auto [slope, intercept] = segment_model.get_floating_point_segment(segment_data.front().first);
        
//         // 获取当前段的第一个键
//         KEY_TYPE first_key = segment_data.front().first;

//         intercept = -static_cast<int>(std::round(slope * first_key));

//         // // 创建并存储最后的 Segment 对象
//         // underlying_segs.emplace_back(static_cast<typename Segment<KEY_TYPE, data_type>::Floating>(slope), 
//         //                              static_cast<int32_t>(intercept), first_key, segment_keys);

//         // 检查最后的段是否至少有两个不同的键
//         if (segment_keys.size() > 1) {
//             // 创建并存储最后的 Segment 对象
//             underlying_segs.emplace_back(static_cast<typename Segment<KEY_TYPE, data_type>::Floating>(slope), 
//                                          static_cast<int32_t>(intercept), first_key, segment_keys);
//         } else {
//             std::cerr << "Warning: Last segment too small or invalid, skipping." << std::endl;
//         }

//     }

//     return underlying_segs;
// }

// template<typename KEY_TYPE, typename data_type>
// std::vector<fiting::alex::AlexDataNode<KEY_TYPE, data_type>> segment_data_with_optimal_model(
// // std::vector<LIPPWHH<KEY_TYPE, data_type>> segment_data_with_optimal_model(
//     std::vector<std::pair<KEY_TYPE, data_type>>& underlying_data, size_t epsilon) {
  
//     // 初始化返回的段向量
//     std::vector<fiting::alex::AlexDataNode<KEY_TYPE, data_type>> underlying_segs;
//     // std::vector<LIPPWHH<KEY_TYPE, data_type>> underlying_segs;
    
//     // 初始化最优线性分段模型，使用给定的 epsilon 值
//     typename fiting::internal::OptimalPiecewiseLinearModel<KEY_TYPE, size_t> opt_model(static_cast<size_t>(epsilon));
    
//     // 临时保存段内数据
//     std::vector<std::pair<KEY_TYPE, data_type>> segment_data;

//     // 对象池，存放已经创建的 LIPPWHH 对象
//     // std::vector<LIPPWHH<KEY_TYPE, data_type>> node_pool;
//     std::vector<fiting::alex::AlexDataNode<KEY_TYPE, data_type>> node_pool;
//     // 动态调整reserve的初始值，初始设置为0，当segment_data有数据后进行调整
//     size_t segment_reserve_size = 0;
    
//     // 遍历输入数据 underlying_data
//     for (size_t i = 0; i < underlying_data.size(); ++i) {
//         KEY_TYPE& x = underlying_data[i].first;  // 当前数据点的键
//         const data_type& y = underlying_data[i].second;  // 当前数据点的值

//         // 动态调整reserve的大小
//         if (segment_data.empty()) {
//             segment_reserve_size = std::max(segment_reserve_size, static_cast<size_t>(underlying_data.size() / 100));
//             segment_data.reserve(segment_reserve_size);
//         }

//         // 添加当前点到分段模型中，x 是键，i 是下标
//         if (!opt_model.add_point(x, i)) {

//             // std::cout << "before get_segment " << std::endl;

//                         // // 当前点无法加入现有段，生成一个新段
//                         // auto segment_model = opt_model.get_segment();
                        
//                         // // 计算当前段的斜率和截距
//                         // auto [slope, intercept] = segment_model.get_floating_point_segment(segment_data.front().first);

//                         // fiting::alex::LinearModel<KEY_TYPE> linear_model(slope, intercept);

//             // std::cout << "INININ: Segment " << underlying_segs.size() + 1 << " - Loading bulk data" << std::endl;

//             // // 将临时段数据加载到 AlexDataNode
//             // // fiting::alex::AlexDataNode<KEY_TYPE, data_type> node;
//             // LIPPWHH<KEY_TYPE, data_type> node;
//             // // node.bulk_load(segment_data.data(), segment_data.size(), &linear_model);
//             // node.bulk_load(segment_data.data(), segment_data.size());

//             // std::cout << "OUTOUTOUT: Segment " << underlying_segs.size() + 1 << " - Bulk load complete" << std::endl;

//             // // 将 AlexDataNode 添加到返回结果中
//             // underlying_segs.push_back(std::move(node));
//             // LIPPWHH<KEY_TYPE, data_type>* node;
//             fiting::alex::AlexDataNode<KEY_TYPE, data_type>* node;
//             if (!node_pool.empty()) {
//                 node = &node_pool.back();  // 重用对象池中的对象
//                 node_pool.pop_back();
//             } else {
//                 // node = new LIPPWHH<KEY_TYPE, data_type>();  // 分配新对象
//                 node = new fiting::alex::AlexDataNode<KEY_TYPE, data_type>();  // 分配新对象
//             }

//             node->bulk_load(segment_data.data(), segment_data.size());
//             underlying_segs.push_back(std::move(*node));

//             // std::cout << "before clear " << std::endl;
            
//             // 清空当前段的数据并重置分段模型
//             segment_data.clear();
//             opt_model.reset();

//             // 调整下次段的reserve大小，避免太大或太小
//             segment_reserve_size = std::max(segment_reserve_size, segment_data.size());
//             segment_data.shrink_to_fit();  // 释放多余内存
            
//             // 重新开始新的段
//             opt_model.add_point(x, i);
//             segment_data.push_back({x, y});

//             node_pool.push_back(std::move(*node));
//         } else {
//             // 如果当前点可以加入段内，则继续添加到临时数据中
//             segment_data.push_back({x, y});
//         }
//     }
    
//     // 处理最后一个段
//     if (!segment_data.empty()) {
//         // auto segment_model = opt_model.get_segment();
//         // auto [slope, intercept] = segment_model.get_floating_point_segment(segment_data.front().first);
        
//         // fiting::alex::LinearModel<KEY_TYPE> linear_model(slope, intercept);

//         // // std::cout << "INININ: Final Segment - Loading bulk data" << std::endl;

//         // // 将临时段数据加载到 AlexDataNode
//         // // fiting::alex::AlexDataNode<KEY_TYPE, data_type> node;
//         // LIPPWHH<KEY_TYPE, data_type> node;
//         // // node.bulk_load(segment_data.data(), segment_data.size(), &linear_model);
//         // node.bulk_load(segment_data.data(), segment_data.size());
//         // // node.set_firstkey_int(segment_data.front().first);

//         // // std::cout << "OUTOUTOUT: Final Segment - Bulk load complete" << std::endl;

//         // // 将 AlexDataNode 添加到返回结果中
//         // underlying_segs.push_back(std::move(node));

// //直接打开上面的整体 注释掉下面的整体就可以改回来

//                      // auto segment_model = opt_model.get_segment();        
        
//         // LIPPWHH<KEY_TYPE, data_type>* node;
//         fiting::alex::AlexDataNode<KEY_TYPE, data_type>* node;
//         if (!node_pool.empty()) {
//             node = &node_pool.back();  // 重用对象池中的对象
//             node_pool.pop_back();
//         } else {
//             // node = new LIPPWHH<KEY_TYPE, data_type>();  // 分配新对象
//             node = new fiting::alex::AlexDataNode<KEY_TYPE, data_type>();  // 分配新对象
//         }

//         node->bulk_load(segment_data.data(), segment_data.size());
//         underlying_segs.push_back(std::move(*node));

//         node_pool.push_back(std::move(*node));
//     }
    

//     // //下面的方法是先记录alex的分段情况，然后用alex的分段方式来分段构建alexdatanode
//     // // 读取 first_key.txt 文件中的键值
//     // std::vector<KEY_TYPE> first_keys;
//     // std::ifstream infile("first_keys.txt");
//     // if (!infile.is_open()) {
//     //     std::cerr << "Failed to open first_keys.txt" << std::endl;
//     //     return {};
//     // }

//     // std::string line;
    
//     // // 跳过第一行注释内容
//     // std::getline(infile, line);
    
//     // // 从第二行开始读取键值
//     // KEY_TYPE key;
//     // while (infile >> key) {
//     //     first_keys.push_back(key);
//     // }
//     // infile.close();

//     // // 初始化返回的段向量
//     // std::vector<fiting::alex::AlexDataNode<KEY_TYPE, data_type>> underlying_segs;

//     // // 检查数据是否为空
//     // if (underlying_data.empty() || first_keys.empty()) {
//     //     return underlying_segs;
//     // }

//     // // // 确保 first_keys 是排序的
//     // // std::sort(first_keys.begin(), first_keys.end());

//     // // 当前段的起点
//     // size_t segment_start = 0;

//     // // 遍历 first_keys，创建每一个 AlexDataNode
//     // for (size_t i = 1; i < first_keys.size(); ++i) {
//     //     KEY_TYPE current_key = first_keys[i];
//     //     std::vector<std::pair<KEY_TYPE, data_type>> segment_data;

//     //     // 收集属于当前段的数据
//     //     while (segment_start < underlying_data.size() && underlying_data[segment_start].first < current_key) {
//     //         segment_data.push_back(underlying_data[segment_start]);
//     //         ++segment_start;
//     //     }

//     //     if (!segment_data.empty()) {
//     //         // 使用分段的数据创建 AlexDataNode
//     //         fiting::alex::AlexDataNode<KEY_TYPE, data_type> node;
//     //         node.bulk_load(segment_data.data(), segment_data.size());

//     //         // 将 AlexDataNode 添加到返回结果中
//     //         underlying_segs.push_back(std::move(node));
//     //     }
//     // }

//     // // 处理最后一个段
//     // if (segment_start < underlying_data.size()) {
//     //     std::vector<std::pair<KEY_TYPE, data_type>> segment_data;

//     //     // 收集最后一段的数据
//     //     while (segment_start < underlying_data.size()) {
//     //         segment_data.push_back(underlying_data[segment_start]);
//     //         ++segment_start;
//     //     }

//     //     if (!segment_data.empty()) {
//     //         // 使用分段的数据创建 AlexDataNode
//     //         fiting::alex::AlexDataNode<KEY_TYPE, data_type> node;
//     //         node.bulk_load(segment_data.data(), segment_data.size());

//     //         // 将 AlexDataNode 添加到返回结果中
//     //         underlying_segs.push_back(std::move(node));
//     //     }
//     // }

//     return underlying_segs;
// }




template<typename Fin, typename Fout>
size_t make_segmentation(size_t n, size_t epsilon, Fin in, Fout out) {
    if (n == 0)
        return 0;

    using X = typename std::invoke_result_t<Fin, size_t>::first_type;
    using Y = typename std::invoke_result_t<Fin, size_t>::second_type;
    size_t c = 0;
    auto p = in(0);

    OptimalPiecewiseLinearModel<X, Y> opt(epsilon);
    opt.add_point(p.first, p.second);

    for (size_t i = 1; i < n; ++i) {
        auto next_p = in(i);
        if (next_p.first == p.first)
            continue;
        p = next_p;
        if (!opt.add_point(p.first, p.second)) {
            out(opt.get_segment());
            opt.add_point(p.first, p.second);
            ++c;
        }
    }

    out(opt.get_segment());
    return ++c;
}

template<typename Fin, typename Fout>
size_t make_segmentation_par(size_t n, size_t epsilon, Fin in, Fout out) {
    auto parallelism = std::min(std::min(omp_get_num_procs(), omp_get_max_threads()), 20);
    auto chunk_size = n / parallelism;
    auto c = 0ull;

    if (parallelism == 1 || n < 1ull << 15)
        return make_segmentation(n, epsilon, in, out);

    using X = typename std::invoke_result_t<Fin, size_t>::first_type;
    using Y = typename std::invoke_result_t<Fin, size_t>::second_type;
    using canonical_segment = typename OptimalPiecewiseLinearModel<X, Y>::CanonicalSegment;
    std::vector<std::vector<canonical_segment>> results(parallelism);

    #pragma omp parallel for reduction(+:c) num_threads(parallelism)
    for (auto i = 0; i < parallelism; ++i) {
        auto first = i * chunk_size;
        auto last = i == parallelism - 1 ? n : first + chunk_size;
        if (first > 0) {
            for (; first < last; ++first)
                if (in(first).first != in(first - 1).first)
                    break;
            if (first == last)
                continue;
        }

        auto in_fun = [in, first](auto j) { return in(first + j); };
        auto out_fun = [&results, i](const auto &cs) { results[i].emplace_back(cs); };
        results[i].reserve(chunk_size / (epsilon > 0 ? epsilon * epsilon : 16));
        c += make_segmentation(last - first, epsilon, in_fun, out_fun);
    }

    for (auto &v : results)
        for (auto &cs : v)
            out(cs);

    return c;
}

template<typename RandomIt>
auto make_segmentation(RandomIt first, RandomIt last, size_t epsilon) {
    using key_type = typename RandomIt::value_type;
    using canonical_segment = typename OptimalPiecewiseLinearModel<key_type, size_t>::CanonicalSegment;
    using pair_type = typename std::pair<key_type, size_t>;

    size_t n = std::distance(first, last);
    std::vector<canonical_segment> out;
    out.reserve(epsilon > 0 ? n / (epsilon * epsilon) : n / 16);

    auto in_fun = [first](auto i) { return pair_type(first[i], i); };
    auto out_fun = [&out](const auto &cs) { out.push_back(cs); };
    make_segmentation(n, epsilon, in_fun, out_fun);

    return out;
}

}