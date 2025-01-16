Now you should run one of the following depending on your shell
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
source /share/apps/python/miniconda3.9/etc/profile.d/conda.csh
  0%|          | 0/100 [00:00<?, ?it/s]  1%|          | 1/100 [00:00<01:01,  1.60it/s]  2%|▏         | 2/100 [00:02<01:53,  1.16s/it]  3%|▎         | 3/100 [00:02<01:07,  1.45it/s]  5%|▌         | 5/100 [00:04<01:17,  1.23it/s]  6%|▌         | 6/100 [00:04<00:58,  1.61it/s]  7%|▋         | 7/100 [00:04<00:43,  2.12it/s]  8%|▊         | 8/100 [00:04<00:33,  2.74it/s] 10%|█         | 10/100 [00:04<00:22,  4.05it/s] 11%|█         | 11/100 [00:04<00:18,  4.71it/s] 13%|█▎        | 13/100 [00:04<00:14,  6.13it/s] 14%|█▍        | 14/100 [00:05<00:13,  6.26it/s] 16%|█▌        | 16/100 [00:05<00:10,  7.84it/s] 17%|█▋        | 17/100 [00:07<00:50,  1.63it/s] 18%|█▊        | 18/100 [00:09<01:17,  1.06it/s] 19%|█▉        | 19/100 [00:09<00:59,  1.35it/s] 20%|██        | 20/100 [00:09<00:45,  1.74it/s] 21%|██        | 21/100 [00:09<00:35,  2.22it/s] 23%|██▎       | 23/100 [00:10<00:22,  3.43it/s] 24%|██▍       | 24/100 [00:10<00:26,  2.85it/s] 24%|██▍       | 24/100 [00:10<00:33,  2.24it/s]
Traceback (most recent call last):
  File "/rcfs/projects/mantis_shrimp/mantis_shrimp/process.py", line 106, in <module>
    DF = pd.read_csv(f'/rcfs/projects/mantis_shrimp/mantis_shrimp/data/url_to_filenames/{survey}/wget{index}.csv')
  File "/people/enge625/.local/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 912, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/people/enge625/.local/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 577, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/people/enge625/.local/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1407, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/people/enge625/.local/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1661, in _make_engine
    self.handles = get_handle(
  File "/people/enge625/.local/lib/python3.9/site-packages/pandas/io/common.py", line 859, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: '/rcfs/projects/mantis_shrimp/mantis_shrimp/data/url_to_filenames/galex/wget4324000.csv'
