import numpy as np


def compute_new_bbox(idx, bbox_min, bbox_max):
    midpoint = (bbox_max - bbox_min) // 2 + bbox_min
    # Compute global block bounding box
    cur_bbox_min = bbox_min.copy()
    cur_bbox_max = midpoint.copy()
    if idx & 1:
        cur_bbox_min[0] = midpoint[0]
        cur_bbox_max[0] = bbox_max[0]
    if (idx >> 1) & 1:
        cur_bbox_min[1] = midpoint[1]
        cur_bbox_max[1] = bbox_max[1]
    if (idx >> 2) & 1:
        cur_bbox_min[2] = midpoint[2]
        cur_bbox_max[2] = bbox_max[2]

    return cur_bbox_min, cur_bbox_max


# Partitions points using an octree scheme
# returns points in local coordinates (block) and octree structure as an 8 bit integer (right to left order)
def split_octree(points, bbox_min, bbox_max):
    ret_points = [[] for x in range(8)]
    midpoint = (bbox_max - bbox_min) // 2
    global_bboxes = [compute_new_bbox(i, bbox_min, bbox_max) for i in range(8)]
    # Translate into local block coordinates
    # Use local block bounding box
    local_bboxes = [(np.zeros(3), x[1] - x[0]) for x in global_bboxes]
    for point in points:
        location = 0
        if point[0] >= midpoint[0]:
            location |= 0b001
        if point[1] >= midpoint[1]:
            location |= 0b010
        if point[2] >= midpoint[2]:
            location |= 0b100
        ret_points[location].append(point - np.pad(global_bboxes[location][0], [0, len(point) - 3]))
    binstr = 0b00000000
    for i, rp in enumerate(ret_points):
        if len(rp) > 0:
            binstr |= (0b00000001 << i)

    return [np.vstack(rp) for rp in ret_points if len(rp) > 0], binstr, local_bboxes


# Returns list of blocks and octree structure as a list of 8 bit integers
# Recursive octree partitioning function that is slow for high number of points (> 500k)
def partition_octree_rec(points, bbox_min, bbox_max, level): #到这里
    if len(points) == 0:
        return [points], None
    if level == 0:
        return [points], None
    bbox_min = np.asarray(bbox_min)
    bbox_max = np.asarray(bbox_max)
    ret_points, binstr, bboxes = split_octree(points, bbox_min, bbox_max)
    result = [partition_octree(rp, bbox[0], bbox[1], level - 1) for rp, bbox in zip(ret_points, bboxes)]
    blocks = [subblock for block_res in result for subblock in block_res[0] if len(subblock) > 0]
    new_binstr = [binstr] + [subbinstr for block_res in result if block_res[1] is not None for subbinstr in block_res[1]]
    return blocks, new_binstr


# Returns list of blocks and octree structure as a list of 8 bit integers
# This version should be much faster than the fully recursive version
# Example: longdress_vox10 73.6s for recursive versus 7.6s for iterative
# However, this assumes that bbox_min is [0, 0, 0] and bbox_max is a power of 2
#这个函数，对points原始点整除64，去重，得到202个点，再将202个点综合考虑zyx从小到大排序，得到block_ids_unique，相当于将points按照大小分了202个类，
#再将points坐标对64取余数，作为local坐标，再根据从原始点到block_ids_unique的映射，把local坐标放到对应的202个类中的一个里面，所有原始点都这么处理，得到blocks。
def partition_octree(points, bbox_min, bbox_max, level):
    points = np.asarray(points)
    if len(points) == 0:#points.shape: (757691, 6)
        return [points], None
    if level == 0: #4
        return [points], None
    bbox_min = np.asarray(bbox_min)
    np.testing.assert_array_equal(bbox_min, [0, 0, 0])
    bbox_max = np.asarray(bbox_max) #[1024, 1024, 1024]
    geo_level = int(np.ceil(np.log2(np.max(bbox_max)))) #10
    assert geo_level >= level
    block_size = 2 ** (geo_level - level) #64

    # Compute partitions for each point
    block_ids = points[:, :3] // block_size
    block_ids = block_ids.astype(np.uint32)
    block_ids_unique, block_idx, block_len = np.unique(block_ids, return_inverse=True, return_counts=True, axis=0)#block_ids_unique.shape: (202, 3)
    #block_ids_unique：去重后的 block_idx：旧的元素在新元素中的索引 block_len：新元素在旧数组中出现的次数
    # Interleave coordinate bits to reorder by octree block order
    sort_key = []
    for x, y, z in block_ids_unique:
        zip_params = [f'{v:0{geo_level - level}b}' for v in [z, y, x]] #展开成6位的二进制字符串表示 ['000100', '001101', '001001']
        sort_key.append(''.join(i + j + k for i, j, k in zip(*zip_params))) #三个字符串中，依次各取一个字符，组合在一起，比如第一个是'000'，合在一起就是000000011110000011
    # print("zip_params:",zip_params) #['000100', '001101', '001001']
    # print("sort_key[-1]:",sort_key[-1]) # 000000011110000011
    sort_idx = np.argsort(sort_key) #从小到大排列 综合考虑zyx的大小，z优先一点点
    block_ids_unique = block_ids_unique[sort_idx]
    block_len = block_len[sort_idx]
    # invert permutation
    inv_sort_idx = np.zeros_like(sort_idx)
    inv_sort_idx[sort_idx] = np.arange(sort_idx.size)
    block_idx = inv_sort_idx[block_idx] #block_idx：旧的元素在重新排序后的block_ids_unique中的索引

    # Translate points into local block coordinates
    local_refs = np.pad(block_ids_unique[block_idx] * block_size, [[0, 0], [0, points.shape[1] - 3]]) #右边加3列0
    points_local = points - local_refs #旧元素对64取余，得到本地坐标

    # Group points by block
    blocks = [np.zeros((l, points.shape[1])) for l in block_len] #长度是202
    blocks_last_idx = np.zeros(len(block_len), dtype=np.uint32) #长度是202
    for i, b_idx in enumerate(block_idx):
        blocks[b_idx][blocks_last_idx[b_idx]] = points_local[i] #取余后的local元素按照重新排序后的block_ids_unique的顺序，聚在一起
        blocks_last_idx[b_idx] += 1

    # Build binary string recursively using the block_ids
    _, binstr = partition_octree_rec(block_ids_unique, [0, 0, 0], (2 ** level) * np.array([1, 1, 1]), level) #这里没细看

    return blocks, binstr #blocks里面的坐标的值都是64的余数，但是排列顺序是有讲究的，基本按从小到大的顺序，把原始点分成不同的类，然后在类之间进行排序，类内没有排序
#binstr 是block_ids_unique进行八叉树排序后的结构编码

def departition_octree(blocks, binstr_list, bbox_min, bbox_max, level):
    bbox_min = np.asarray(bbox_min)
    bbox_max = np.asarray(bbox_max)

    blocks = [b.copy() for b in blocks]
    binstr_list = binstr_list.copy()
    binstr_idxs = np.zeros(len(binstr_list), dtype=np.uint8) #长度67
    children_counts = np.zeros(len(binstr_list), dtype=np.uint32)

    binstr_list_idx = 0
    block_idx = 0
    cur_level = 1

    bbox_stack = [(bbox_min, bbox_max)]
    parents_stack = []

    while block_idx < len(blocks):
        child_found = False
        # Find next child at current level
        while binstr_list[binstr_list_idx] != 0 and not child_found:
            if (binstr_list[binstr_list_idx] & 1) == 1:
                v = binstr_idxs[binstr_list_idx]
                cur_bbox = compute_new_bbox(v, *bbox_stack[-1]) #cur_bbox: (array([0, 0, 0]), array([512, 512, 512]))
                if cur_level == level:
                    # Leaf node: decode current block
                    blocks[block_idx] = blocks[block_idx] + np.pad(cur_bbox[0], [0, blocks[block_idx].shape[1] - 3])
                    # print(f'Read block {block_idx} at binstr {binstr_list_idx} ({cur_bbox})')
                    block_idx += 1
                else:
                    # print(f'Child found at idx {binstr_idxs[binstr_list_idx]} for binstr {binstr_list_idx}')
                    # Non leaf child: stop reading current binstr
                    child_found = True

            binstr_list[binstr_list_idx] >>= 1
            binstr_idxs[binstr_list_idx] += 1

        if child_found:
            # Child found: descend octree
            bbox_stack.append(cur_bbox)
            parents_stack.append(binstr_list_idx)
            for i in range(len(parents_stack)):
                children_counts[parents_stack[i]] += 1
            # Go to child
            cur_level += 1
            binstr_list_idx += children_counts[parents_stack[-1]]
            # print(f'Descend to {binstr_list_idx}')
        else:
            # No children left: ascend octree
            binstr_list_idx = parents_stack.pop()
            cur_level -= 1
            bbox_stack.pop()
            # print(f'Ascend to {binstr_list_idx}')

    return blocks

