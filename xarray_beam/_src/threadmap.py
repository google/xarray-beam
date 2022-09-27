# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Beam Map transforms that execute in parallel using a thread pool.

This can be a good idea for IO bound tasks, but generally should be avoided for
CPU bound tasks, especially CPU bound tasks that do not release the Python GIL.

They can be used as drop-in substitutes for the corresponding Beam transforms:
- beam.Map -> ThreadMap
- beam.MapTuple -> ThreadMapTuple
- beam.FlatMap -> FlatThreadMap
- beam.FlatMapTuple -> FlatThreadMapTuple

By default, 16 threads are used per task. This can be adjusted via the
`num_threads` keyword argument.
"""
import concurrent.futures
import functools

import apache_beam as beam


class ThreadDoFn(beam.DoFn):
  """A DoFn that executes inputs in a ThreadPool."""

  def __init__(self, func, num_threads):
    self.func = func
    self.num_threads = num_threads

  def setup(self):
    self.executor = concurrent.futures.ThreadPoolExecutor(self.num_threads)

  def teardown(self):
    self.executor.shutdown()

  def process(self, element, *args, **kwargs):
    futures = []
    for x in element:
      futures.append(self.executor.submit(self.func, x, *args, **kwargs))
    for future in futures:
      yield future.result()


class FlatThreadDoFn(ThreadDoFn):

  def process(self, element, *args, **kwargs):
    for results in super().process(element, *args, **kwargs):
      yield from results


class _ThreadMap(beam.PTransform):
  """Like beam.Map, but executed in a thread-pool."""

  def __init__(self, func, *args, num_threads, **kwargs):
    self.func = func
    self.args = args
    self.kwargs = kwargs
    self.num_threads = num_threads

  def get_dofn(self):
    return ThreadDoFn(self.func, self.num_threads)

  def expand(self, pcoll):
    return (
        pcoll
        | 'BatchElements'
        >> beam.BatchElements(
            min_batch_size=self.num_threads,
            max_batch_size=self.num_threads,
        )
        | 'ParDo' >> beam.ParDo(self.get_dofn(), *self.args, **self.kwargs)
    )


class _ThreadMapTuple(_ThreadMap):
  """Like beam.MapTuple, but executed in a thread-pool."""

  def get_dofn(self):
    func = lambda xs, **kwargs: self.func(*xs, **kwargs)
    return ThreadDoFn(func, self.num_threads)


class _FlatThreadMap(_ThreadMap):
  """Like beam.FlatMap, but executed in a thread-pool."""

  def get_dofn(self):
    return FlatThreadDoFn(self.func, self.num_threads)


class _FlatThreadMapTuple(_ThreadMap):
  """Like beam.FlatMapTuple, but executed in a thread-pool."""

  def get_dofn(self):
    func = lambda xs, **kwargs: self.func(*xs, **kwargs)
    return FlatThreadDoFn(func, self.num_threads)


def _maybe_threaded(beam_transform, thread_transform):
  @functools.wraps(thread_transform)
  def create(func, *args, num_threads=16, **kwargs):
    if num_threads is None:
      return beam_transform(func, *args, **kwargs)
    else:
      return thread_transform(func, *args, num_threads=num_threads, **kwargs)

  return create


# These functions don't use threads if num_threads=None.
ThreadMap = _maybe_threaded(beam.Map, _ThreadMap)
ThreadMapTuple = _maybe_threaded(beam.MapTuple, _ThreadMapTuple)
FlatThreadMap = _maybe_threaded(beam.FlatMap, _FlatThreadMap)
FlatThreadMapTuple = _maybe_threaded(beam.FlatMapTuple, _FlatThreadMapTuple)
