

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tqdm import tqdm
from multiprocessing import Pool
import os
from os import listdir
from glob import glob
import argparse

from toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, VOTDataset, NFSDataset, VOTLTDataset, GOT10kDataset
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark
from toolkit.visualization import draw_f1, draw_eao, draw_success_precision
from configs.DataPath import TEST_PATH, SYSTEM

parser = argparse.ArgumentParser(description='tracking evaluation')
parser.add_argument('--dataset', '-d', default='UAV', type=str, help='dataset name')
parser.add_argument('--num', '-n', default=4, type=int, help='number of thread to evaluate')
parser.add_argument('--tracker_prefix', '-t', default='SiamDCA', type=str, help='tracker name')
parser.add_argument('--show_video_level', '-s', dest='show_video_level', action='store_true')
parser.set_defaults(show_video_level=False)
args = parser.parse_args()


def main():
    dataset = args.dataset
    root = TEST_PATH[dataset]
    tracker_dir = os.path.join('results', dataset)

    trackers = glob(os.path.join('results', dataset, args.tracker_prefix + '*'))
    if SYSTEM == 'Linux':
        trackers = [x.split('/')[-1] for x in trackers]
    else:
        trackers = [x.split('\\')[-1] for x in trackers]
    # 在debug模式下运行，方便找出对比算法结果中的异常box文件以及在调参时取出中间结果
    # dataset = 'OTB100'
    # trackers = ['DaSiamRPN', 'Ocean-off', 'SiamCAR', 'SiamBAN', 'SiamFC++', 'SiamGAT', 'ATOM', 'DiMP-50', 'SiamDCA']
    # trackers = listdir('./results/' + dataset)

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    if 'OTB' in dataset:
        dataset = OTBDataset(dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)

        # 检查问题出现在哪个tracker的哪些序列上
        # ret = benchmark.eval_success(trackers)
        # pret = benchmark.eval_precision(trackers)

        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='evaluate success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='evaluate precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)

        # 最大显示数调大，以便观察超参数搜索时的结果
        benchmark.show_result(success_ret, precision_ret, show_video_level=args.show_video_level, max_num=60)

        # 取出某个tracker在每个序列上的结果
        # a0 = list(precision_ret.keys())
        # index = 8
        # print(a0[index])
        # a1 = precision_ret[a0[index]]
        # a2 = list(a1.values())
        # a3 = [v[20] for v in a2]
        # a3 = np.array(a3).reshape((-1, 1))
        # a4 = list(a1.keys())

        # 可视化
        for attr, videos in dataset.attr.items():
            draw_success_precision(success_ret,
                                   name=dataset.name,
                                   videos=videos,
                                   attr=attr,
                                   precision_ret=precision_ret)

    elif dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset = VOTDataset(dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        benchmark = EAOBenchmark(dataset, tags=dataset.tags)

        a = ar_benchmark.eval(trackers)
        b = benchmark.eval(trackers)
        ar_benchmark.show_result(a, b, show_video_level=args.show_video_level)
        draw_eao(b)

        ar_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                                                trackers), desc='evaluate ar', total=len(trackers), ncols=100):
                ar_result.update(ret)
        # benchmark.show_result(ar_result)

        eao_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                                                trackers), desc='evaluate eao', total=len(trackers), ncols=100):
                eao_result.update(ret)
        # benchmark.show_result(eao_result)

        ar_benchmark.show_result(ar_result, eao_result, show_video_level=args.show_video_level, show_num=50)
        draw_eao(eao_result)

    elif 'UAV' in dataset:
        dataset = UAVDataset(dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        # a = benchmark.eval_success(trackers)
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='evaluate success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='evaluate precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)

        # a0 = list(precision_ret.keys())
        # index = 0
        # print(a0[index])
        # a1 = precision_ret[a0[index]]
        # a2 = list(a1.values())
        # a3 = [v[20] for v in a2]
        # a3 = np.array(a3).reshape((-1, 1))
        # a4 = list(a1.keys())

        benchmark.show_result(success_ret, precision_ret, show_video_level=args.show_video_level)
        for attr, videos in dataset.attr.items():
            draw_success_precision(success_ret,
                                   name=dataset.name,
                                   videos=videos,
                                   attr=attr,
                                   precision_ret=precision_ret)

    elif 'LaSOT' == dataset:
        dataset = LaSOTDataset(dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        # success_ret = benchmark.eval_success(trackers)
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='evaluate success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='evaluate precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                                                trackers), desc='evaluate norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                              show_video_level=args.show_video_level)
        draw_success_precision(success_ret,
                               name=dataset.name,
                               videos=dataset.attr['ALL'],
                               attr='ALL',
                               precision_ret=precision_ret,
                               norm_precision_ret=norm_precision_ret)

    elif 'NFS' in dataset:
        dataset = NFSDataset(dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        ret = benchmark.eval_success(trackers)
        pret = benchmark.eval_precision(trackers)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='evaluate success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='evaluate precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
        for attr, videos in dataset.attr.items():
            draw_success_precision(success_ret,
                                   name=dataset.name,
                                   videos=videos,
                                   attr=attr,
                                   precision_ret=precision_ret)

    elif 'VOT2018-LT' == dataset:
        dataset = VOTLTDataset(dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                                                trackers), desc='evaluate f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result,
                              show_video_level=args.show_video_level)
        draw_f1(f1_result)


if __name__ == '__main__':
    main()
