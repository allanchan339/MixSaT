import os


def find_gpus(num_of_cards_needed=4):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >~/.tmp_free_gpus')
    # If there is no ~ in the path, return the path unchanged
    with open(os.path.expanduser('~/.tmp_free_gpus'), 'r') as lines_txt:
        frees = lines_txt.readlines()
        idx_freeMemory_pair = [(idx, int(x.split()[2]))
                               for idx, x in enumerate(frees)]
    idx_freeMemory_pair.sort(reverse=True)  # 0号卡经常有人抢，让最后一张卡在下面的sort中优先
    idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
    usingGPUs = [str(idx_memory_pair[0]) for idx_memory_pair in
                 idx_freeMemory_pair[:num_of_cards_needed]]
    usingGPUs = ','.join(usingGPUs)
    for pair in idx_freeMemory_pair[:num_of_cards_needed]:
        print(f'GPU {pair[0]} is free ：{pair[1]/1024:.1f}GB')
    return usingGPUs


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(
        num_of_cards_needed=4)  # must before `import torch`
