import math
import heapq
from copy import deepcopy

class Frame():
    def __init__(
        self,
        obj,
        frame_id,
        score
    ):
        self.score_decayed = score
        self.obj = obj
        self.frame_id = frame_id
        self.score = score

    def __lt__(self, other):
        a = (
            self.score_decayed,
            self.score,
            self.frame_id
        )
        b = (
            other.score_decayed,
            other.score,
            other.frame_id
        )
        return a < b

class Memory():
    def __init__(
        self,
        memory_len = 1,
        fix_first_frame = False,
        fix_last_frame = False,
        memory_decay_ratio = 20,
        memory_decay_type = 'cos'
    ):
        self.memory_len = memory_len - fix_last_frame - fix_first_frame
        assert self.memory_len >= 0
        self.memory = []
        self.fix_first_frame = fix_first_frame
        self.first_frame = None
        self.fix_last_frame = fix_last_frame
        self.last_frame = None
        self.score_decay = memory_decay_ratio

        if memory_decay_ratio:
            if memory_decay_type == 'cos':
                self.score_decay_table = [max(0, math.cos(x/memory_decay_ratio)) for x in range(15)]
            elif memory_decay_type == 'linear':
                self.score_decay_table = [max(0, 1-x/memory_decay_ratio) for x in range(int(memory_decay_ratio))]
            elif memory_decay_type == 'ellipse':
                self.score_decay_table = [max(0, (1-(x/memory_decay_ratio)**2)**0.5) for x in range(int(memory_decay_ratio))]
            elif memory_decay_type == 'exp':
                self.score_decay_table = [max(0, math.exp(-x/memory_decay_ratio)) for x in range(15)]
            elif memory_decay_type == 'constant':
                self.score_decay_table = [1] * 15


    def _get_score_decay_ratio(self, x):
        if not self.score_decay:
            return 1
        elif x < len(self.score_decay_table):
            return self.score_decay_table[x]
        else:
            return 0

    def update_memory(self, frame):
        if self.fix_first_frame and self.first_frame is None:
            self.first_frame = frame
            return

        if self.fix_last_frame:
            if self.last_frame is not None:
                heapq.heappush(self.memory, self.last_frame)
            self.last_frame = frame
        else:
            heapq.heappush(self.memory, frame)

        for i in range(len(self.memory)):
            score_decay_ratio = self._get_score_decay_ratio(frame.frame_id - self.memory[i].frame_id)
            score_decayed = score_decay_ratio * self.memory[i].score
            self.memory[i].score_decayed = score_decayed
        heapq.heapify(self.memory)
        if len(self.memory) > self.memory_len:
            heapq.heappop(self.memory)

    def get_memory(self):
        memory = deepcopy(self.memory)
        if self.first_frame is not None:
            memory.append(self.first_frame)

        if self.last_frame is not None:
            memory.append(self.last_frame)

        return memory

    def clear_memory(self):
        self.last_frame = None
        self.memory = []