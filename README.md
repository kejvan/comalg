# Comprehensive programming algorithms guide

## Table of contents

### **Core algorithm patterns**

1. [Two pointers](#1-two-pointers)
2. [Sliding window](#2-sliding-window)
3. [Binary search](#3-binary-search)
4. [Prefix sum](#4-prefix-sum)
5. [Sorting algorithms](#5-sorting-algorithms)
6. [Kadane's algorithm (maximum subarray)](#6-kadanes-algorithm-maximum-subarray)
7. [Dutch national flag algorithm](#7-dutch-national-flag-algorithm)
8. [Cyclic sort](#8-cyclic-sort)
9. [Monotonic stack/queue](#9-monotonic-stackqueue)
10. [Difference array](#10-difference-array)

### **String processing algorithms**

11. [String pattern matching](#11-string-pattern-matching)
12. [String manipulation](#12-string-manipulation)

### **Graph & matrix algorithms**

13. [Matrix algorithms](#13-matrix-algorithms)
14. [Graph traversal](#14-graph-traversal)

### **Hash-based algorithms**

15. [Frequency analysis](#15-frequency-analysis)
16. [Sum mapping](#16-sum-mapping)
17. [Duplicate detection](#17-duplicate-detection)

### **Advanced techniques**

18. [Array reconstruction](#18-array-reconstruction)
19. [Design data structures](#19-design-data-structures)

---

## 1. Two pointers

### Basic idea

Use two pointers to traverse data structures from different positions, reducing time complexity from O(n²) to O(n) for many problems.

### Types:

1. **Same direction**: both pointers move in the same direction
2. **Opposite direction**: pointers move towards each other
3. **Fast-slow**: one pointer moves faster than the other

### Basic two pointers (same target)

**Concept**: use two pointers starting from different positions to find pairs or triplets that satisfy certain conditions.

```python
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1

    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []
```

**LeetCode problems:**

- [1. Two Sum](https://leetcode.com/problems/two-sum/)
- [167. Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)
- [15. 3Sum](https://leetcode.com/problems/3sum/)
- [16. 3Sum Closest](https://leetcode.com/problems/3sum-closest/)
- [18. 4Sum](https://leetcode.com/problems/4sum/)
- [259. 3Sum Smaller](https://leetcode.com/problems/3sum-smaller/)
- [611. Valid Triangle Number](https://leetcode.com/problems/valid-triangle-number/)
- [923. 3Sum With Multiplicity](https://leetcode.com/problems/3sum-with-multiplicity/)

### Opposite direction two pointers

**Concept**: start from both ends and move towards center, useful for palindromes and container problems.

```python
def container_with_most_water(height):
    left, right = 0, len(height) - 1
    max_area = 0

    while left < right:
        width = right - left
        area = min(height[left], height[right]) * width
        max_area = max(max_area, area)

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area
```

**LeetCode problems:**

- [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/)
- [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)
- [125. Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)
- [680. Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/)
- [881. Boats to Save People](https://leetcode.com/problems/boats-to-save-people/)
- [977. Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/)
- [986. Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/)
- [88. Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)

### Fast-slow pointers (Floyd's algorithm)

**Concept**: one pointer moves twice as fast as the other, used for cycle detection and finding middle elements.

```python
def find_duplicate(nums):
    # Phase 1: Find intersection point
    slow = fast = nums[0]

    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break

    # Phase 2: Find entrance to cycle
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]

    return slow
```

**LeetCode problems:**

- [141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)
- [142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)
- [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)
- [202. Happy Number](https://leetcode.com/problems/happy-number/)
- [876. Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)
- [234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)
- [457. Circular Array Loop](https://leetcode.com/problems/circular-array-loop/)
- [143. Reorder List](https://leetcode.com/problems/reorder-list/)

---

## 2. Sliding window

### Basic idea

Maintain a window (subarray) and slide it across the array. Efficient for problems involving contiguous subarrays.

### Types:

1. **Fixed size window**: window size remains constant
2. **Variable size window**: window size changes based on conditions

### Fixed size window

**Concept**: maintain a window of fixed size k and slide it across the array.

```python
def max_average_subarray(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum

    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)

    return max_sum / k
```

**LeetCode problems:**

- [643. Maximum Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/)
- [1456. Maximum Number of Vowels in a Substring of Given Length](https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/)
- [1004. Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/)
- [567. Permutation in String](https://leetcode.com/problems/permutation-in-string/)
- [1343. Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold](https://leetcode.com/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/)
- [1052. Grumpy Bookstore Owner](https://leetcode.com/problems/grumpy-bookstore-owner/)
- [1423. Maximum Points You Can Obtain from Cards](https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/)
- [480. Sliding Window Median](https://leetcode.com/problems/sliding-window-median/)

### Variable size window

**Concept**: expand and contract the window based on conditions using two pointers.

```python
def longest_substring_without_repeating(s):
    char_set = set()
    left = 0
    max_length = 0

    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)

    return max_length
```

**LeetCode problems:**

- [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
- [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)
- [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)
- [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)
- [904. Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/)
- [992. Subarrays with K Different Integers](https://leetcode.com/problems/subarrays-with-k-different-integers/)
- [1358. Number of Substrings Containing All Three Characters](https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/)

---

## 3. Binary search

### Basic idea

Divide and conquer approach to search in sorted arrays. Time complexity: O(log n).

### Template:

```python
def binary_search(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

### Basic binary search

**Concept**: standard binary search in sorted array.

**LeetCode problems:**

- [704. Binary Search](https://leetcode.com/problems/binary-search/)
- [35. Search Insert Position](https://leetcode.com/problems/search-insert-position/)
- [278. First Bad Version](https://leetcode.com/problems/first-bad-version/)
- [374. Guess Number Higher or Lower](https://leetcode.com/problems/guess-number-higher-or-lower/)
- [744. Find Smallest Letter Greater Than Target](https://leetcode.com/problems/find-smallest-letter-greater-than-target/)
- [441. Arranging Coins](https://leetcode.com/problems/arranging-coins/)
- [1351. Count Negative Numbers in a Sorted Matrix](https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/)
- [1539. Kth Missing Positive Number](https://leetcode.com/problems/kth-missing-positive-number/)

### Binary search on answer

**Concept**: binary search on the answer range when the answer has a monotonic property.

```python
def koko_eating_bananas(piles, h):
    def can_finish(speed):
        time = 0
        for pile in piles:
            time += (pile + speed - 1) // speed  # Ceiling division
        return time <= h

    left, right = 1, max(piles)

    while left < right:
        mid = left + (right - left) // 2
        if can_finish(mid):
            right = mid
        else:
            left = mid + 1

    return left
```

**LeetCode problems:**

- [69. Sqrt(x)](https://leetcode.com/problems/sqrtx/)
- [367. Valid Perfect Square](https://leetcode.com/problems/valid-perfect-square/)
- [1011. Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)
- [875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)
- [410. Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/)
- [1482. Minimum Number of Days to Make m Bouquets](https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/)
- [1552. Magnetic Force Between Two Balls](https://leetcode.com/problems/magnetic-force-between-two-balls/)
- [1283. Find the Smallest Divisor Given a Threshold](https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/)

### Binary search in rotated arrays

**Concept**: modified binary search for rotated sorted arrays.

```python
def search_rotated_array(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid

        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

**LeetCode problems:**

- [33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [81. Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)
- [153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)
- [154. Find Minimum in Rotated Sorted Array II](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/)
- [4. Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)
- [1095. Find in Mountain Array](https://leetcode.com/problems/find-in-mountain-array/)
- [540. Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/)
- [702. Search in a Sorted Array of Unknown Size](https://leetcode.com/problems/search-in-a-sorted-array-of-unknown-size/)

### Binary search for first/last occurrence

**Concept**: find the first or last occurrence of a target in sorted array with duplicates.

```python
def find_first_last_position(nums, target):
    def find_first():
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
        return left if left < len(nums) and nums[left] == target else -1

    def find_last():
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return right if right >= 0 and nums[right] == target else -1

    first = find_first()
    last = find_last()
    return [first, last]
```

**LeetCode problems:**

- [34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
- [162. Find Peak Element](https://leetcode.com/problems/find-peak-element/)
- [852. Peak Index in a Mountain Array](https://leetcode.com/problems/peak-index-in-a-mountain-array/)
- [1095. Find in Mountain Array](https://leetcode.com/problems/find-in-mountain-array/)
- [278. First Bad Version](https://leetcode.com/problems/first-bad-version/)
- [1231. Divide Chocolate](https://leetcode.com/problems/divide-chocolate/)
- [528. Random Pick with Weight](https://leetcode.com/problems/random-pick-with-weight/)
- [1060. Missing Element in Sorted Array](https://leetcode.com/problems/missing-element-in-sorted-array/)

### Binary search in 2D arrays

**Concept**: apply binary search in 2D matrices with sorted properties.

```python
def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False

    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1

    while left <= right:
        mid = left + (right - left) // 2
        mid_value = matrix[mid // cols][mid % cols]

        if mid_value == target:
            return True
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1

    return False
```

**LeetCode problems:**

- [74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)
- [240. Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/)
- [378. Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)
- [1337. The K Weakest Rows in a Matrix](https://leetcode.com/problems/the-k-weakest-rows-in-a-matrix/)
- [1283. Find the Smallest Divisor Given a Threshold](https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/)
- [1201. Ugly Number III](https://leetcode.com/problems/ugly-number-iii/)
- [1044. Longest Duplicate Substring](https://leetcode.com/problems/longest-duplicate-substring/)
- [1102. Path With Maximum Minimum Value](https://leetcode.com/problems/path-with-maximum-minimum-value/)

---

## 4. Prefix sum

### Basic idea

Precompute cumulative sums to answer range sum queries in O(1) time after O(n) preprocessing.

### Basic prefix sum

**Concept**: store cumulative sums to quickly calculate subarray sums.

```python
class PrefixSum:
    def __init__(self, nums):
        self.prefix = [0]
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)

    def range_sum(self, left, right):
        return self.prefix[right + 1] - self.prefix[left]
```

**LeetCode problems:**

- [303. Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/)
- [724. Find Pivot Index](https://leetcode.com/problems/find-pivot-index/)
- [1480. Running Sum of 1d Array](https://leetcode.com/problems/running-sum-of-1d-array/)
- [1732. Find the Highest Altitude](https://leetcode.com/problems/find-the-highest-altitude/)
- [1588. Sum of All Odd Length Subarrays](https://leetcode.com/problems/sum-of-all-odd-length-subarrays/)
- [1893. Check if All the Integers in a Range Are Covered](https://leetcode.com/problems/check-if-all-the-integers-in-a-range-are-covered/)
- [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)
- [1991. Find the Middle Index in Array](https://leetcode.com/problems/find-the-middle-index-in-array/)

### Prefix sum with HashMap

**Concept**: use hashmap to store prefix sums and find subarrays with specific properties.

```python
def subarray_sum_equals_k(nums, k):
    count = 0
    prefix_sum = 0
    sum_count = {0: 1}  # Handle subarrays starting from index 0

    for num in nums:
        prefix_sum += num
        if prefix_sum - k in sum_count:
            count += sum_count[prefix_sum - k]
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1

    return count
```

**LeetCode problems:**

- [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
- [525. Contiguous Array](https://leetcode.com/problems/contiguous-array/)
- [974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/)
- [523. Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/)
- [930. Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/)
- [1442. Count Triplets That Can Form Two Arrays of Equal XOR](https://leetcode.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/)
- [437. Path Sum III](https://leetcode.com/problems/path-sum-iii/)
- [1074. Number of Submatrices That Sum to Target](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/)

### 2D prefix sum

**Concept**: extend prefix sum to 2D matrices for range sum queries.

```python
class Matrix2DPrefixSum:
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            return

        m, n = len(matrix), len(matrix[0])
        self.prefix = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                self.prefix[i][j] = (matrix[i-1][j-1] +
                                   self.prefix[i-1][j] +
                                   self.prefix[i][j-1] -
                                   self.prefix[i-1][j-1])

    def sum_region(self, row1, col1, row2, col2):
        return (self.prefix[row2+1][col2+1] -
                self.prefix[row1][col2+1] -
                self.prefix[row2+1][col1] +
                self.prefix[row1][col1])
```

**LeetCode problems:**

- [304. Range Sum Query 2D - Immutable](https://leetcode.com/problems/range-sum-query-2d-immutable/)
- [1314. Matrix Block Sum](https://leetcode.com/problems/matrix-block-sum/)
- [1139. Largest 1-Bordered Square](https://leetcode.com/problems/largest-1-bordered-square/)
- [1277. Count Square Submatrices with All Ones](https://leetcode.com/problems/count-square-submatrices-with-all-ones/)
- [1504. Count Submatrices With All Ones](https://leetcode.com/problems/count-submatrices-with-all-ones/)
- [1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold](https://leetcode.com/problems/maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/)
- [1738. Find Kth Largest XOR Coordinate Value](https://leetcode.com/problems/find-kth-largest-xor-coordinate-value/)
- [1863. Sum of All Subset XOR Totals](https://leetcode.com/problems/sum-of-all-subset-xor-totals/)

---

## 5. Sorting algorithms

### Merge sort

**Concept**: divide array into halves, sort recursively, then merge. Time: O(n log n), Space: O(n).

```python
def merge_sort(nums):
    if len(nums) <= 1:
        return nums

    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

**LeetCode problems:**

- [88. Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)
- [148. Sort List](https://leetcode.com/problems/sort-list/)
- [315. Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)
- [493. Reverse Pairs](https://leetcode.com/problems/reverse-pairs/)
- [327. Count of Range Sum](https://leetcode.com/problems/count-of-range-sum/)
- [1305. All Elements in Two Binary Search Trees](https://leetcode.com/problems/all-elements-in-two-binary-search-trees/)
- [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)
- [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

### Quick sort & partitioning

**Concept**: choose pivot, partition array, recursively sort subarrays. Average: O(n log n).

```python
def quick_sort(nums, low=0, high=None):
    if high is None:
        high = len(nums) - 1

    if low < high:
        pi = partition(nums, low, high)
        quick_sort(nums, low, pi - 1)
        quick_sort(nums, pi + 1, high)

def partition(nums, low, high):
    pivot = nums[high]
    i = low - 1

    for j in range(low, high):
        if nums[j] <= pivot:
            i += 1
            nums[i], nums[j] = nums[j], nums[i]

    nums[i + 1], nums[high] = nums[high], nums[i + 1]
    return i + 1

def quickselect(nums, k):
    """Find kth largest element"""
    def partition(left, right):
        pivot = nums[right]
        i = left
        for j in range(left, right):
            if nums[j] >= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[right] = nums[right], nums[i]
        return i

    left, right = 0, len(nums) - 1
    while True:
        pos = partition(left, right)
        if pos == k - 1:
            return nums[pos]
        elif pos > k - 1:
            right = pos - 1
        else:
            left = pos + 1
```

**LeetCode problems:**

- [75. Sort Colors](https://leetcode.com/problems/sort-colors/)
- [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)
- [973. K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)
- [324. Wiggle Sort II](https://leetcode.com/problems/wiggle-sort-ii/)
- [912. Sort an Array](https://leetcode.com/problems/sort-an-array/)
- [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
- [692. Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/)
- [703. Kth Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/)

### Bucket sort / counting sort

**Concept**: distribute elements into buckets, sort individually. Good for limited range values.

```python
def bucket_sort_frequencies(nums, k):
    """Top K Frequent Elements using bucket sort"""
    count = {}
    for num in nums:
        count[num] = count.get(num, 0) + 1

    # Create buckets
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)

    # Collect top k elements
    result = []
    for i in range(len(buckets) - 1, -1, -1):
        for num in buckets[i]:
            if len(result) < k:
                result.append(num)

    return result
```

**LeetCode problems:**

- [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
- [451. Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/)
- [164. Maximum Gap](https://leetcode.com/problems/maximum-gap/)
- [414. Third Maximum Number](https://leetcode.com/problems/third-maximum-number/)
- [274. H-Index](https://leetcode.com/problems/h-index/)
- [1122. Relative Sort Array](https://leetcode.com/problems/relative-sort-array/)
- [1636. Sort Array by Increasing Frequency](https://leetcode.com/problems/sort-array-by-increasing-frequency/)
- [692. Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/)

---

## 6. Kadane's algorithm (maximum subarray)

### Basic idea

Find maximum sum of contiguous subarray in O(n) time using dynamic programming approach.

### Basic maximum subarray

**Concept**: keep track of maximum sum ending at current position.

```python
def max_subarray(nums):
    max_ending_here = max_so_far = nums[0]

    for i in range(1, len(nums)):
        max_ending_here = max(nums[i], max_ending_here + nums[i])
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far

def max_product_subarray(nums):
    max_prod = min_prod = result = nums[0]

    for i in range(1, len(nums)):
        if nums[i] < 0:
            max_prod, min_prod = min_prod, max_prod

        max_prod = max(nums[i], max_prod * nums[i])
        min_prod = min(nums[i], min_prod * nums[i])
        result = max(result, max_prod)

    return result
```

**LeetCode problems:**

- [53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)
- [152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)
- [918. Maximum Sum Circular Subarray](https://leetcode.com/problems/maximum-sum-circular-subarray/)
- [1186. Maximum Subarray Sum with One Deletion](https://leetcode.com/problems/maximum-subarray-sum-with-one-deletion/)
- [1191. K-Concatenation Maximum Sum](https://leetcode.com/problems/k-concatenation-maximum-sum/)
- [1746. Maximum Subarray Sum After One Operation](https://leetcode.com/problems/maximum-subarray-sum-after-one-operation/)
- [1425. Constrained Subsequence Sum](https://leetcode.com/problems/constrained-subsequence-sum/)
- [1562. Find Latest Group of Size M](https://leetcode.com/problems/find-latest-group-of-size-m/)

### Variations of maximum subarray

**Concept**: apply Kadane's algorithm to stock problems and similar scenarios.

```python
def max_profit(prices):
    """Best Time to Buy and Sell Stock"""
    min_price = float('inf')
    max_profit = 0

    for price in prices:
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price

    return max_profit
```

**LeetCode problems:**

- [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
- [122. Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)
- [123. Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)
- [188. Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)
- [309. Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)
- [714. Best Time to Buy and Sell Stock with Transaction Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)
- [1014. Best Sightseeing Pair](https://leetcode.com/problems/best-sightseeing-pair/)
- [1395. Count Number of Teams](https://leetcode.com/problems/count-number-of-teams/)

---

## 7. Dutch national flag algorithm

### Basic idea

Partition array into three sections using three pointers. Useful for sorting arrays with three distinct values.

### Three-way partitioning

**Concept**: use three pointers to partition array into three sections.

```python
def sort_colors(nums):
    """Sort array of 0s, 1s, and 2s"""
    low = mid = 0
    high = len(nums) - 1

    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
            # Don't increment mid here as we need to check swapped element
```

**LeetCode problems:**

- [75. Sort Colors](https://leetcode.com/problems/sort-colors/)
- [274. H-Index](https://leetcode.com/problems/h-index/)
- [275. H-Index II](https://leetcode.com/problems/h-index-ii/)
- [905. Sort Array By Parity](https://leetcode.com/problems/sort-array-by-parity/)
- [922. Sort Array By Parity II](https://leetcode.com/problems/sort-array-by-parity-ii/)
- [948. Bag of Tokens](https://leetcode.com/problems/bag-of-tokens/)
- [969. Pancake Sorting](https://leetcode.com/problems/pancake-sorting/)
- [1296. Divide Array in Sets of K Consecutive Numbers](https://leetcode.com/problems/divide-array-in-sets-of-k-consecutive-numbers/)

---

## 8. Cyclic sort

### Basic idea

For arrays containing numbers from 1 to n, place each number at its correct index (number i at index i-1).

### Missing number problems

**Concept**: use the fact that numbers should be at specific indices to find missing/duplicate elements.

```python
def cyclic_sort(nums):
    i = 0
    while i < len(nums):
        correct_index = nums[i] - 1
        if nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            i += 1

def find_missing_number(nums):
    n = len(nums)
    # Place each number at its correct position
    i = 0
    while i < n:
        if nums[i] < n and nums[i] != nums[nums[i]]:
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        else:
            i += 1

    # Find missing number
    for i in range(n):
        if nums[i] != i:
            return i
    return n
```

**LeetCode problems:**

- [268. Missing Number](https://leetcode.com/problems/missing-number/)
- [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)
- [41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/)
- [448. Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)
- [442. Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/)
- [645. Set Mismatch](https://leetcode.com/problems/set-mismatch/)
- [1539. Kth Missing Positive Number](https://leetcode.com/problems/kth-missing-positive-number/)
- [1060. Missing Element in Sorted Array](https://leetcode.com/problems/missing-element-in-sorted-array/)

---

## 9. Monotonic stack/queue

### Basic idea

Maintain elements in monotonic order (increasing or decreasing) to efficiently find next/previous greater/smaller elements.

### Monotonic stack

**Concept**: stack maintains elements in monotonic order, useful for next greater element problems.

```python
def next_greater_elements(nums):
    """Next Greater Element II (circular array)"""
    n = len(nums)
    result = [-1] * n
    stack = []

    # Process array twice to handle circular nature
    for i in range(2 * n):
        # Pop elements smaller than current
        while stack and nums[stack[-1]] < nums[i % n]:
            result[stack.pop()] = nums[i % n]

        # Only push indices in first pass
        if i < n:
            stack.append(i)

    return result

def largest_rectangle_in_histogram(heights):
    """Find largest rectangle area in histogram"""
    stack = []
    max_area = 0

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    while stack:
        height = heights[stack.pop()]
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        max_area = max(max_area, height * width)

    return max_area
```

**LeetCode problems:**

- [496. Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)
- [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)
- [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
- [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
- [85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)
- [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)
- [1944. Number of Visible People in a Queue](https://leetcode.com/problems/number-of-visible-people-in-a-queue/)
- [1762. Buildings With an Ocean View](https://leetcode.com/problems/buildings-with-an-ocean-view/)

### Monotonic deque

**Concept**: deque maintains elements in monotonic order, useful for sliding window maximum/minimum.

```python
from collections import deque

def sliding_window_maximum(nums, k):
    """Find maximum in each sliding window of size k"""
    if not nums:
        return []

    dq = deque()  # Store indices
    result = []

    for i in range(len(nums)):
        # Remove indices outside window
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Remove smaller elements from back
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Add to result when window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

**LeetCode problems:**

- [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
- [862. Shortest Subarray with Sum at Least K](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/)
- [1696. Jump Game VI](https://leetcode.com/problems/jump-game-vi/)
- [1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit](https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)
- [1499. Max Value of Equation](https://leetcode.com/problems/max-value-of-equation/)
- [1425. Constrained Subsequence Sum](https://leetcode.com/problems/constrained-subsequence-sum/)
- [907. Sum of Subarray Minimums](https://leetcode.com/problems/sum-of-subarray-minimums/)
- [1130. Minimum Cost Tree From Leaf Values](https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/)

---

## 10. Difference array

### Basic idea

Efficiently handle multiple range update operations on an array. Build difference array where diff[i] = arr[i] - arr[i-1].

### Range update operations

**Concept**: use difference array to apply range updates in O(1) time.

```python
class DifferenceArray:
    def __init__(self, nums):
        self.diff = [0] * len(nums)
        self.diff[0] = nums[0]
        for i in range(1, len(nums)):
            self.diff[i] = nums[i] - nums[i-1]

    def update_range(self, left, right, val):
        """Add val to range [left, right]"""
        self.diff[left] += val
        if right + 1 < len(self.diff):
            self.diff[right + 1] -= val

    def get_array(self):
        """Reconstruct original array"""
        result = [0] * len(self.diff)
        result[0] = self.diff[0]
        for i in range(1, len(self.diff)):
            result[i] = result[i-1] + self.diff[i]
        return result

def corporate_flight_bookings(bookings, n):
    """Range addition using difference array"""
    diff = [0] * (n + 1)

    for first, last, seats in bookings:
        diff[first - 1] += seats
        diff[last] -= seats

    result = []
    current = 0
    for i in range(n):
        current += diff[i]
        result.append(current)

    return result
```

**LeetCode problems:**

- [370. Range Addition](https://leetcode.com/problems/range-addition/)
- [1109. Corporate Flight Bookings](https://leetcode.com/problems/corporate-flight-bookings/)
- [1094. Car Pooling](https://leetcode.com/problems/car-pooling/)
- [598. Range Addition II](https://leetcode.com/problems/range-addition-ii/)
- [1450. Number of Students Doing Homework at a Given Time](https://leetcode.com/problems/number-of-students-doing-homework-at-a-given-time/)
- [1674. Minimum Moves to Make Array Complementary](https://leetcode.com/problems/minimum-moves-to-make-array-complementary/)
- [253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)
- [731. My Calendar II](https://leetcode.com/problems/my-calendar-ii/)

---

## 11. String pattern matching

### Basic idea

Algorithms for finding patterns within strings using various techniques including KMP, rolling hash, and Z-algorithm.

### KMP (Knuth-Morris-Pratt) algorithm

**Concept**: use partial match table to skip unnecessary comparisons in pattern matching.

```python
def str_str(haystack, needle):
    """Find first occurrence of needle in haystack (KMP algorithm)"""
    if not needle:
        return 0

    # Build LPS array
    def build_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1

        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    lps = build_lps(needle)
    i = j = 0

    while i < len(haystack):
        if haystack[i] == needle[j]:
            i += 1
            j += 1

        if j == len(needle):
            return i - j
        elif i < len(haystack) and haystack[i] != needle[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return -1
```

**LeetCode problems:**

- [28. Find the Index of the First Occurrence in a String](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/)
- [214. Shortest Palindrome](https://leetcode.com/problems/shortest-palindrome/)
- [1392. Longest Happy Prefix](https://leetcode.com/problems/longest-happy-prefix/)
- [459. Repeated Substring Pattern](https://leetcode.com/problems/repeated-substring-pattern/)
- [796. Rotate String](https://leetcode.com/problems/rotate-string/)
- [686. Repeated String Match](https://leetcode.com/problems/repeated-string-match/)
- [1668. Maximum Repeating Substring](https://leetcode.com/problems/maximum-repeating-substring/)
- [1147. Longest Chunked Palindrome Decomposition](https://leetcode.com/problems/longest-chunked-palindrome-decomposition/)

### Rolling hash pattern matching

**Concept**: use rolling hash for efficient substring matching and comparison.

```python
def longest_duplicate_substring(s):
    """Find longest duplicate substring using rolling hash"""
    def search(L):
        # Rolling hash using polynomial rolling hash
        base = 26
        mod = 2**63 - 1

        h = 0
        for i in range(L):
            h = (h * base + ord(s[i]) - ord('a')) % mod

        seen = {h}
        base_L = pow(base, L, mod)

        for start in range(1, len(s) - L + 1):
            # Rolling hash: remove leading digit, add trailing digit
            h = (h * base - (ord(s[start - 1]) - ord('a')) * base_L + ord(s[start + L - 1]) - ord('a')) % mod
            if h in seen:
                return start
            seen.add(h)
        return -1

    # Binary search on answer
    left, right = 1, len(s)
    start = -1

    while left <= right:
        L = left + (right - left) // 2
        if search(L) != -1:
            start = search(L)
            left = L + 1
        else:
            right = L - 1

    return s[start:start + right] if start != -1 else ""
```

**LeetCode problems:**

- [1044. Longest Duplicate Substring](https://leetcode.com/problems/longest-duplicate-substring/)
- [718. Maximum Length of Repeated Subarray](https://leetcode.com/problems/maximum-length-of-repeated-subarray/)
- [1062. Longest Repeating Substring](https://leetcode.com/problems/longest-repeating-substring/)
- [1316. Distinct Echo Substrings](https://leetcode.com/problems/distinct-echo-substrings/)
- [1554. Strings Differ by One Character](https://leetcode.com/problems/strings-differ-by-one-character/)
- [1923. Longest Common Subpath](https://leetcode.com/problems/longest-common-subpath/)
- [1713. Minimum Operations to Make a Subsequence](https://leetcode.com/problems/minimum-operations-to-make-a-subsequence/)
- [1948. Delete Duplicate Folders in System](https://leetcode.com/problems/delete-duplicate-folders-in-system/)

---

## 12. String manipulation

### Basic idea

In-place string modifications, transformations, and character operations.

### String reversal and rotation

**Concept**: efficiently reverse strings and detect rotations.

```python
def reverse_words(s):
    """Reverse words in a string"""
    # Remove extra spaces and split
    words = s.strip().split()
    return ' '.join(reversed(words))

def reverse_words_in_string_iii(s):
    """Reverse each word individually"""
    words = s.split(' ')
    return ' '.join(word[::-1] for word in words)

def is_palindrome(s):
    """Check if string is palindrome (ignoring case and non-alphanumeric)"""
    left, right = 0, len(s) - 1

    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True
```

**LeetCode problems:**

- [151. Reverse Words in a String](https://leetcode.com/problems/reverse-words-in-a-string/)
- [557. Reverse Words in a String III](https://leetcode.com/problems/reverse-words-in-a-string-iii/)
- [125. Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)
- [680. Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/)
- [344. Reverse String](https://leetcode.com/problems/reverse-string/)
- [345. Reverse Vowels of a String](https://leetcode.com/problems/reverse-vowels-of-a-string/)
- [917. Reverse Only Letters](https://leetcode.com/problems/reverse-only-letters/)
- [1332. Remove Palindromic Subsequences](https://leetcode.com/problems/remove-palindromic-subsequences/)

### String compression and encoding

**Concept**: compress strings and handle encoding/decoding problems.

```python
def compress(chars):
    """String compression in-place"""
    write = 0
    i = 0

    while i < len(chars):
        char = chars[i]
        count = 1

        # Count consecutive characters
        while i + count < len(chars) and chars[i + count] == char:
            count += 1

        # Write character
        chars[write] = char
        write += 1

        # Write count if > 1
        if count > 1:
            for digit in str(count):
                chars[write] = digit
                write += 1

        i += count

    return write
```

**LeetCode problems:**

- [443. String Compression](https://leetcode.com/problems/string-compression/)
- [6. Zigzag Conversion](https://leetcode.com/problems/zigzag-conversion/)
- [38. Count and Say](https://leetcode.com/problems/count-and-say/)
- [14. Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/)
- [58. Length of Last Word](https://leetcode.com/problems/length-of-last-word/)
- [1071. Greatest Common Divisor of Strings](https://leetcode.com/problems/greatest-common-divisor-of-strings/)
- [394. Decode String](https://leetcode.com/problems/decode-string/)
- [271. Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/)

---

## 13. Matrix algorithms

### Basic idea

Special techniques for 2D array problems including spiral traversal, rotation, and search.

### Matrix traversal patterns

**Concept**: navigate matrices in specific patterns (spiral, diagonal, etc.).

```python
def spiral_order(matrix):
    """Spiral traversal of matrix"""
    if not matrix:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Traverse right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1

        # Traverse down
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1

        if top <= bottom:
            # Traverse left
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1

        if left <= right:
            # Traverse up
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1

    return result
```

**LeetCode problems:**

- [54. Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)
- [59. Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii/)
- [48. Rotate Image](https://leetcode.com/problems/rotate-image/)
- [498. Diagonal Traverse](https://leetcode.com/problems/diagonal-traverse/)
- [1329. Sort the Matrix Diagonally](https://leetcode.com/problems/sort-the-matrix-diagonally/)
- [885. Spiral Matrix III](https://leetcode.com/problems/spiral-matrix-iii/)
- [1914. Cyclically Rotating a Grid](https://leetcode.com/problems/cyclically-rotating-a-grid/)
- [1968. Array With Elements Not Equal to Average of Neighbors](https://leetcode.com/problems/array-with-elements-not-equal-to-average-of-neighbors/)

### Matrix modification

**Concept**: in-place matrix modifications and transformations.

```python
def set_matrix_zeroes(matrix):
    """Set entire row and column to zero if element is zero"""
    if not matrix:
        return

    rows, cols = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][j] == 0 for j in range(cols))
    first_col_zero = any(matrix[i][0] == 0 for i in range(rows))

    # Use first row and column as markers
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0

    # Set zeros based on markers
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0

    # Handle first row and column
    if first_row_zero:
        for j in range(cols):
            matrix[0][j] = 0

    if first_col_zero:
        for i in range(rows):
            matrix[i][0] = 0
```

**LeetCode problems:**

- [73. Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/)
- [289. Game of Life](https://leetcode.com/problems/game-of-life/)
- [48. Rotate Image](https://leetcode.com/problems/rotate-image/)
- [1582. Special Positions in a Binary Matrix](https://leetcode.com/problems/special-positions-in-a-binary-matrix/)
- [1260. Shift 2D Grid](https://leetcode.com/problems/shift-2d-grid/)
- [1380. Lucky Numbers in a Matrix](https://leetcode.com/problems/lucky-numbers-in-a-matrix/)
- [1672. Richest Customer Wealth](https://leetcode.com/problems/richest-customer-wealth/)
- [1779. Find Nearest Point That Has the Same X or Y Coordinate](https://leetcode.com/problems/find-nearest-point-that-has-the-same-x-or-y-coordinate/)

### Matrix search

**Concept**: efficient search in sorted matrices and matrix transformations.

```python
def search_matrix_ii(matrix, target):
    """Search in row-wise and column-wise sorted matrix"""
    if not matrix or not matrix[0]:
        return False

    row, col = 0, len(matrix[0]) - 1

    while row < len(matrix) and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1

    return False
```

**LeetCode problems:**

- [240. Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/)
- [74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)
- [378. Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)
- [1337. The K Weakest Rows in a Matrix](https://leetcode.com/problems/the-k-weakest-rows-in-a-matrix/)
- [1351. Count Negative Numbers in a Sorted Matrix](https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/)
- [1572. Matrix Diagonal Sum](https://leetcode.com/problems/matrix-diagonal-sum/)
- [1304. Find N Unique Integers Sum up to Zero](https://leetcode.com/problems/find-n-unique-integers-sum-up-to-zero/)
- [1295. Find Numbers with Even Number of Digits](https://leetcode.com/problems/find-numbers-with-even-number-of-digits/)

---

## 14. Graph traversal

### Basic idea

Use DFS/BFS with hash sets to track visited nodes and solve graph-based problems.

### Island problems (DFS/BFS)

**Concept**: use DFS or BFS to find connected components in grid problems.

```python
def num_islands(grid):
    """Count number of islands using DFS with visited set"""
    if not grid:
        return 0

    visited = set()
    count = 0

    def dfs(i, j):
        if (i, j) in visited or i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == '0':
            return

        visited.add((i, j))
        # Visit all 4 directions
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1' and (i, j) not in visited:
                dfs(i, j)
                count += 1

    return count
```

**LeetCode problems:**

- [200. Number of Islands](https://leetcode.com/problems/number-of-islands/)
- [417. Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)
- [130. Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)
- [733. Flood Fill](https://leetcode.com/problems/flood-fill/)
- [1905. Count Sub Islands](https://leetcode.com/problems/count-sub-islands/)
- [1254. Number of Closed Islands](https://leetcode.com/problems/number-of-closed-islands/)
- [694. Number of Distinct Islands](https://leetcode.com/problems/number-of-distinct-islands/)
- [1219. Path with Maximum Gold](https://leetcode.com/problems/path-with-maximum-gold/)

### Path finding

**Concept**: use BFS for shortest path problems and DFS for all paths problems.

```python
def shortest_path_binary_matrix(grid):
    """Shortest path in binary matrix using BFS"""
    if not grid or grid[0][0] == 1 or grid[-1][-1] == 1:
        return -1

    n = len(grid)
    if n == 1:
        return 1

    from collections import deque
    queue = deque([(0, 0, 1)])  # (row, col, distance)
    visited = {(0, 0)}

    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    while queue:
        row, col, dist = queue.popleft()

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if (new_row == n-1 and new_col == n-1):
                return dist + 1

            if (0 <= new_row < n and 0 <= new_col < n and
                grid[new_row][new_col] == 0 and (new_row, new_col) not in visited):
                visited.add((new_row, new_col))
                queue.append((new_row, new_col, dist + 1))

    return -1
```

**LeetCode problems:**

- [1091. Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/)
- [542. 01 Matrix](https://leetcode.com/problems/01-matrix/)
- [994. Rotting Oranges](https://leetcode.com/problems/rotting-oranges/)
- [1162. As Far from Land as Possible](https://leetcode.com/problems/as-far-from-land-as-possible/)
- [934. Shortest Bridge](https://leetcode.com/problems/shortest-bridge/)
- [1926. Nearest Exit from Entrance in Maze](https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/)
- [1730. Shortest Path to Get Food](https://leetcode.com/problems/shortest-path-to-get-food/)
- [1263. Minimum Moves to Move a Box to Their Target Location](https://leetcode.com/problems/minimum-moves-to-move-a-box-to-their-target-location/)

---

## 15. Frequency analysis

### Basic idea

Use HashMap to count frequencies of elements for solving various problems.

### Character/element counting

**Concept**: count frequencies to solve anagram, substring, and grouping problems.

```python
def majority_element(nums):
    """Find majority element (appears > n/2 times)"""
    count = {}
    for num in nums:
        count[num] = count.get(num, 0) + 1
        if count[num] > len(nums) // 2:
            return num
    return -1

def top_k_frequent(nums, k):
    """Find k most frequent elements"""
    from collections import Counter
    import heapq

    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

def group_anagrams(strs):
    """Group strings that are anagrams"""
    anagram_map = {}

    for s in strs:
        # Sort characters as key
        key = ''.join(sorted(s))
        if key not in anagram_map:
            anagram_map[key] = []
        anagram_map[key].append(s)

    return list(anagram_map.values())
```

**LeetCode problems:**

- [169. Majority Element](https://leetcode.com/problems/majority-element/)
- [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
- [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)
- [451. Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/)
- [1636. Sort Array by Increasing Frequency](https://leetcode.com/problems/sort-array-by-increasing-frequency/)
- [383. Ransom Note](https://leetcode.com/problems/ransom-note/)
- [242. Valid Anagram](https://leetcode.com/problems/valid-anagram/)
- [1207. Unique Number of Occurrences](https://leetcode.com/problems/unique-number-of-occurrences/)

### Frequency-based algorithms

**Concept**: use frequency maps for sliding window and substring problems.

```python
def find_anagrams(s, p):
    """Find all anagrams of p in s"""
    from collections import Counter

    if len(p) > len(s):
        return []

    p_count = Counter(p)
    window_count = Counter()
    result = []

    for i in range(len(s)):
        # Add current character
        window_count[s[i]] += 1

        # Remove character outside window
        if i >= len(p):
            if window_count[s[i - len(p)]] == 1:
                del window_count[s[i - len(p)]]
            else:
                window_count[s[i - len(p)]] -= 1

        # Check if current window is anagram
        if window_count == p_count:
            result.append(i - len(p) + 1)

    return result
```

**LeetCode problems:**

- [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)
- [567. Permutation in String](https://leetcode.com/problems/permutation-in-string/)
- [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
- [30. Substring with Concatenation of All Words](https://leetcode.com/problems/substring-with-concatenation-of-all-words/)
- [1002. Find Common Characters](https://leetcode.com/problems/find-common-characters/)
- [1046. Last Stone Weight](https://leetcode.com/problems/last-stone-weight/)
- [692. Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/)
- [609. Find Duplicate File in System](https://leetcode.com/problems/find-duplicate-file-in-system/)

---

## 16. Sum mapping

### Basic idea

Use HashMap to store complements and solve sum-based problems efficiently.

### Two sum patterns

**Concept**: use HashMap to find pairs that sum to target.

```python
def two_sum(nums, target):
    """Find two numbers that add up to target"""
    num_map = {}

    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i

    return []

def four_sum_count(nums1, nums2, nums3, nums4):
    """Count tuples with sum = 0"""
    sum_map = {}

    # Store all possible sums of nums1 and nums2
    for a in nums1:
        for b in nums2:
            sum_ab = a + b
            sum_map[sum_ab] = sum_map.get(sum_ab, 0) + 1

    count = 0
    # Check if -(c + d) exists in sum_map
    for c in nums3:
        for d in nums4:
            target = -(c + d)
            count += sum_map.get(target, 0)

    return count
```

**LeetCode problems:**

- [1. Two Sum](https://leetcode.com/problems/two-sum/)
- [454. 4Sum II](https://leetcode.com/problems/4sum-ii/)
- [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
- [525. Contiguous Array](https://leetcode.com/problems/contiguous-array/)
- [974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/)
- [523. Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/)
- [930. Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/)
- [437. Path Sum III](https://leetcode.com/problems/path-sum-iii/)

### Cumulative sum with HashMap

**Concept**: use prefix sums with HashMap to find subarrays with target properties.

```python
def subarray_sum(nums, k):
    """Count subarrays with sum = k"""
    count = 0
    prefix_sum = 0
    sum_count = {0: 1}  # Handle subarrays starting from index 0

    for num in nums:
        prefix_sum += num
        if prefix_sum - k in sum_count:
            count += sum_count[prefix_sum - k]
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1

    return count
```

**LeetCode problems:**

- [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
- [325. Maximum Size Subarray Sum Equals k](https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/)
- [525. Contiguous Array](https://leetcode.com/problems/contiguous-array/)
- [1074. Number of Submatrices That Sum to Target](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/)
- [1248. Count Number of Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/)
- [974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/)
- [930. Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/)
- [1442. Count Triplets That Can Form Two Arrays of Equal XOR](https://leetcode.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/)

---

## 17. Duplicate detection

### Basic idea

Use HashSet for O(1) lookup to efficiently detect duplicates and unique elements.

### Basic duplicate detection

**Concept**: use HashSet to find duplicates, missing elements, and unique values.

```python
def contains_duplicate(nums):
    """Check if array contains any duplicates"""
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

def contains_nearby_duplicate(nums, k):
    """Check if there are duplicates within distance k"""
    seen = set()
    for i, num in enumerate(nums):
        if num in seen:
            return True
        seen.add(num)
        if len(seen) > k:
            seen.remove(nums[i - k])
    return False

def single_number(nums):
    """Find the single number in array where every other appears twice"""
    seen = set()
    for num in nums:
        if num in seen:
            seen.remove(num)
        else:
            seen.add(num)
    return seen.pop()
```

**LeetCode problems:**

- [217. Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)
- [219. Contains Duplicate II](https://leetcode.com/problems/contains-duplicate-ii/)
- [136. Single Number](https://leetcode.com/problems/single-number/)
- [268. Missing Number](https://leetcode.com/problems/missing-number/)
- [448. Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-array/)
- [41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/)
- [442. Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/)
- [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)

### Set operations

**Concept**: use HashSet for mathematical set operations like intersection, union, and difference.

```python
def intersection(nums1, nums2):
    """Find intersection of two arrays"""
    set1 = set(nums1)
    return list(set1 & set(nums2))

def intersection_with_duplicates(nums1, nums2):
    """Find intersection with duplicates allowed"""
    from collections import Counter
    count1 = Counter(nums1)
    result = []

    for num in nums2:
        if count1[num] > 0:
            result.append(num)
            count1[num] -= 1

    return result
```

**LeetCode problems:**

- [349. Intersection of Two Arrays](https://leetcode.com/problems/intersection-of-two-arrays/)
- [350. Intersection of Two Arrays II](https://leetcode.com/problems/intersection-of-two-arrays-ii/)
- [448. Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)
- [1995. Count Special Quadruplets](https://leetcode.com/problems/count-special-quadruplets/)
- [1213. Intersection of Three Sorted Arrays](https://leetcode.com/problems/intersection-of-three-sorted-arrays/)
- [1002. Find Common Characters](https://leetcode.com/problems/find-common-characters/)
- [1346. Check If N and Its Double Exist](https://leetcode.com/problems/check-if-n-and-its-double-exist/)
- [1748. Sum of Unique Elements](https://leetcode.com/problems/sum-of-unique-elements/)

---

## 18. Array reconstruction

### Basic idea

Use array indices as hash keys to find missing/duplicate elements without extra space.

### In-place array modification

**Concept**: modify array without using extra space, often using indices as hash keys.

```python
def remove_element(nums, val):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
    return slow

def rotate_array(nums, k):
    """Rotate array to the right by k steps"""
    k %= len(nums)
    reverse(nums, 0, len(nums) - 1)
    reverse(nums, 0, k - 1)
    reverse(nums, k, len(nums) - 1)

def reverse(nums, start, end):
    while start < end:
        nums[start], nums[end] = nums[end], nums[start]
        start += 1
        end -= 1
```

**LeetCode problems:**

- [27. Remove Element](https://leetcode.com/problems/remove-element/)
- [26. Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
- [80. Remove Duplicates from Sorted Array II](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/)
- [283. Move Zeroes](https://leetcode.com/problems/move-zeroes/)
- [189. Rotate Array](https://leetcode.com/problems/rotate-array/)
- [41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/)
- [448. Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)
- [442. Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/)

### Index-based algorithms

**Concept**: use array indices as hash keys to solve problems in O(1) space.

```python
def first_missing_positive(nums):
    n = len(nums)

    # Place each positive number i at index i-1
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]

    # Find first missing positive
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    return n + 1
```

**LeetCode problems:**

- [41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/)
- [448. Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)
- [442. Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/)
- [645. Set Mismatch](https://leetcode.com/problems/set-mismatch/)
- [268. Missing Number](https://leetcode.com/problems/missing-number/)
- [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)
- [1539. Kth Missing Positive Number](https://leetcode.com/problems/kth-missing-positive-number/)
- [1426. Counting Elements](https://leetcode.com/problems/counting-elements/)

---

## 19. Design data structures

### Basic idea

Implement data structures using HashMap as the underlying storage mechanism.

### Cache designs

**Concept**: implement LRU/LFU caches using HashMap + Doubly Linked List.

```python
class LRUCache:
    """Least Recently Used Cache using HashMap + Doubly Linked List"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> node

        # Dummy head and tail nodes
        self.head = ListNode(0, 0)
        self.tail = ListNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add_to_head(node)
            return node.val
        return -1

    def put(self, key, value):
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_head(node)
        else:
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                last_node = self.tail.prev
                self._remove(last_node)
                del self.cache[last_node.key]

            # Add new node
            new_node = ListNode(key, value)
            self._add_to_head(new_node)
            self.cache[key] = new_node

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

class ListNode:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None
```

**LeetCode problems:**

- [146. LRU Cache](https://leetcode.com/problems/lru-cache/)
- [460. LFU Cache](https://leetcode.com/problems/lfu-cache/)
- [380. Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/)
- [381. Insert Delete GetRandom O(1) - Duplicates allowed](https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/)
- [705. Design HashSet](https://leetcode.com/problems/design-hashset/)
- [706. Design HashMap](https://leetcode.com/problems/design-hashmap/)
- [1396. Design Underground System](https://leetcode.com/problems/design-underground-system/)
- [1244. Design A Leaderboard](https://leetcode.com/problems/design-a-leaderboard/)

### Advanced data structure designs

**Concept**: design complex data structures with multiple operations.

```python
class TimeMap:
    """Time-based key-value store"""

    def __init__(self):
        self.store = {}  # key -> list of (timestamp, value)

    def set(self, key, value, timestamp):
        if key not in self.store:
            self.store[key] = []
        self.store[key].append((timestamp, value))

    def get(self, key, timestamp):
        if key not in self.store:
            return ""

        values = self.store[key]
        left, right = 0, len(values) - 1
        result = ""

        # Binary search for largest timestamp <= given timestamp
        while left <= right:
            mid = (left + right) // 2
            if values[mid][0] <= timestamp:
                result = values[mid][1]
                left = mid + 1
            else:
                right = mid - 1

        return result
```

**LeetCode problems:**

- [981. Time Based Key-Value Store](https://leetcode.com/problems/time-based-key-value-store/)
- [895. Maximum Frequency Stack](https://leetcode.com/problems/maximum-frequency-stack/)
- [432. All O`one Data Structure](https://leetcode.com/problems/all-oone-data-structure/)
- [1472. Design Browser History](https://leetcode.com/problems/design-browser-history/)
- [622. Design Circular Queue](https://leetcode.com/problems/design-circular-queue/)
- [641. Design Circular Deque](https://leetcode.com/problems/design-circular-deque/)
- [1603. Design Parking System](https://leetcode.com/problems/design-parking-system/)
- [1381. Design a Stack With Increment Operation](https://leetcode.com/problems/design-a-stack-with-increment-operation/)

---

## Algorithm selection guidelines

### When to use each algorithm:

1. **Two pointers**: use for pairs/triplets in sorted arrays or palindrome problems
2. **Sliding window**: use for contiguous subarray/substring problems
3. **Binary search**: use for sorted data or when answer has monotonic property
4. **Prefix sum**: use for range sum queries or cumulative operations
5. **Sorting**: use when order matters or finding kth element
6. **Kadane's**: use for maximum/minimum subarray sum problems
7. **Dutch national flag**: use for partitioning with 3 values
8. **Cyclic sort**: use when array contains numbers 1 to n
9. **Monotonic stack/queue**: use for next/previous greater/smaller elements
10. **Difference array**: use for multiple range update operations
11. **String pattern matching**: use for substring search and pattern problems
12. **String manipulation**: use for string transformations and character operations
13. **Matrix algorithms**: use for 2D grid traversal and manipulation
14. **Graph traversal**: use for connected components and path finding
15. **Frequency analysis**: use for counting and anagram problems
16. **Sum mapping**: use for target sum problems with HashMap
17. **Duplicate detection**: use for finding unique/duplicate elements
18. **Array reconstruction**: use for in-place modifications without extra space
19. **Design data structures**: use for implementing custom data structures

### Time complexity summary:

**Core algorithms:**

- Two pointers: O(n)
- Sliding window: O(n)
- Binary search: O(log n)
- Prefix sum: O(n) preprocessing, O(1) query
- Sorting: O(n log n)
- Kadane's algorithm: O(n)
- Monotonic stack: O(n)

**String/hash algorithms:**

- String pattern matching: O(n + m) for KMP
- HashMap operations: O(1) average
- Frequency counting: O(n)

**Graph/matrix algorithms:**

- Graph traversal: O(V + E)
- Matrix operations: O(m × n)

### Space complexity summary:

**Core algorithms:**

- Two pointers: O(1)
- Sliding window: O(1) to O(k)
- Binary search: O(1)
- Prefix sum: O(n)
- Sorting: O(1) to O(n)

**String/hash algorithms:**

- HashMap: O(n) for storage
- String algorithms: O(1) to O(n)

**Graph/matrix algorithms:**

- Graph traversal: O(V) for visited set
- Matrix algorithms: O(1) to O(m × n)

---

**Practice strategy**: start with two pointers and sliding window, master binary search and prefix sum, then progress to advanced techniques like monotonic stack and specialized string/graph algorithms. Focus on understanding the underlying patterns rather than memorizing solutions.
