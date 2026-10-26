"""20 ms WebRTC VAD with pre-roll. Partial output is a replaceable hypothesis."""
from collections import deque

FRAME = 640
BYTES_PER_SECOND = 32000


class Segmenter:
    def __init__(self, vad, silence_ms=1000, partial_ms=1000, max_ms=15000):
        self.vad = vad
        self.silence_frames = silence_ms // 20
        self.partial_frames = partial_ms // 20
        self.max_frames = max_ms // 20
        self.reset()

    def reset(self):
        self.pre = deque(maxlen=15)
        self.votes = deque(maxlen=10)
        self.frames = []
        self.voiced = 0
        self.quiet = 0
        self.last_partial = 0
        self.elapsed = 0

    def finish(self):
        pcm = b''.join(self.frames) if self.voiced >= 13 else None
        self.reset()
        return pcm

    def feed(self, frame):
        if len(frame) != FRAME:
            raise ValueError('frame_size')
        voice = self.vad.is_speech(frame, 16000)
        self.elapsed += 1
        events = []
        if not self.frames:
            self.pre.append((frame, voice))
            self.votes.append(voice)
            if len(self.votes) == 10 and sum(self.votes) >= 6:
                self.frames = [f for f, _ in self.pre]
                self.voiced = sum(v for _, v in self.pre)
                self.quiet = 0
                events.append(('speech', None))
            return events
        self.frames.append(frame)
        self.voiced += int(voice)
        self.quiet = 0 if voice else self.quiet + 1
        if self.quiet >= self.silence_frames or len(self.frames) >= self.max_frames:
            pcm = self.finish()
            if pcm:
                events.append(('final', pcm))
            else:
                events.append(('discarded', None))
        elif self.voiced >= 13 and len(self.frames) - self.last_partial >= self.partial_frames:
            self.last_partial = len(self.frames)
            events.append(('partial', b''.join(self.frames)))
        return events
