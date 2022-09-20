# pyinstaller -wF Customized_command_recorder.py
# GQ, 2021/02/20, rev.1.0, MIT

import PySimpleGUI as sg
import os
from datetime import datetime
from pathlib import Path
import speech_recognition as sr

from pydub import AudioSegment
from pydub.playback import play
from pydub.silence import detect_leading_silence

import collections
import contextlib
import sys
import wave
# pip install webrtcvad-wheels
import webrtcvad


def trim_mid_and_save(path, required_duration=1000):
    sound = AudioSegment.from_file(path, format="wav")
    start_trim = detect_leading_silence(sound, sound.dBFS - 16)
    end_trim = detect_leading_silence(sound.reverse(), sound.dBFS - 16)

    duration = len(sound)
    trimmed_sound = sound[start_trim:duration - end_trim]
    if len(trimmed_sound) > required_duration:
        trimmed_sound_start = (len(trimmed_sound) - required_duration) // 2
        trimmed_sound = trimmed_sound[trimmed_sound_start:trimmed_sound_start + required_duration]
    else:
        trimmed_sound_silent = (required_duration - len(trimmed_sound)) // 2
        second_of_silence = AudioSegment.silent(duration=trimmed_sound_silent, frame_rate=16000)
        trimmed_sound = second_of_silence + trimmed_sound + second_of_silence

    path = f'{path[:-4]}_trim_mid.wav'
    trimmed_sound.export(path, format="wav")


def trim_and_save(path, required_duration=1000):
    sound = AudioSegment.from_file(path, format="wav")
    start_trim = detect_leading_silence(sound, sound.dBFS - 16)
    end_trim = detect_leading_silence(sound.reverse(), sound.dBFS - 16)

    duration = len(sound)
    trimmed_sound = sound[start_trim:duration - end_trim]
    if len(trimmed_sound) > required_duration:
        trimmed_sound_start = (len(trimmed_sound) - required_duration) // 2
        trimmed_sound = trimmed_sound[trimmed_sound_start:trimmed_sound_start + required_duration]

    path = f'{path[:-4]}_trim.wav'
    trimmed_sound.export(path, format="wav")


# VAD sub===============================================
def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


# VAD main=========================================
def vad_and_save(path, aggressive=3):
    audio, sample_rate = read_wave(path)
    vad = webrtcvad.Vad(aggressive)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    for i, segment in enumerate(segments):
        path = f'{path[:-4]}_vad_{i}.wav'
        write_wave(path, segment, sample_rate)


# Record==========================================
def record_and_save(path):
    # obtain audio from the microphone
    r = sr.Recognizer()
    m = sr.Microphone(sample_rate=16000)
    with m as source:
        # audio = r.listen(source, phrase_time_limit=2)
        # r.adjust_for_ambient_noise(source, duration=0.5)
        window['-TEXT_STATUS-'].update('正在录音', text_color='blue')
        window.refresh()
        audio = r.listen(source)

    cwd_path = Path(path)
    path_file_name = cwd_path / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.wav"

    # write audio to a WAV file
    with open(path_file_name, "wb") as f:
        f.write(audio.get_wav_data())


sg.theme('DarkAmber')  # Add a touch of color
# All the stuff inside your window.
layout = [[sg.T('请先选择保存目录')],
          [sg.In(default_text=os.getcwd(), key='-TEXT_CWD-', enable_events=True),
           sg.FolderBrowse('浏览', initial_folder=os.getcwd())],
          [sg.Button('录音', key='-BTN_RCD-'), sg.T('', size=(40, 1), text_color='red', key='-TEXT_STATUS-')],
          # [sg.Text('_' * 50)],
          [sg.Listbox(values=[file for file in os.listdir('.') if file.endswith(".wav")],
                      select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, size=(50, 20),
                      key='-LIST_WAV-', enable_events=True)],
          [sg.Button('播放', key='-BTN_PLAY-'), sg.Button('删除', key='-BTN_DEL-'),
           sg.Button('语音激活过滤', key='-BTN_VAD-'), sg.Button('去除头尾静音', key='-BTN_TRIM-'),
           sg.Button('头尾静音居中', key='-BTN_TRIM_MID-')]]

cmd_audio = None
selected_path = None

# Create the Window
window = sg.Window('Customized Command Recorder', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
        break
    print(event, values)

    if event == '-TEXT_CWD-':
        if values['-TEXT_CWD-'] == '':
            window['-TEXT_STATUS-'].update('提示：请先选择保存目录')
        else:
            window['-TEXT_STATUS-'].update('')

    if event == '-BTN_RCD-':
        # window['-TEXT_STATUS-'].update('正在录音', text_color='blue')
        # window.refresh()
        record_and_save(values['-TEXT_CWD-'])
        window['-TEXT_STATUS-'].update('')

    if event == '-LIST_WAV-' and values['-LIST_WAV-'] != '':
        selected_audio = True
        selected_path = Path(values['-TEXT_CWD-']) / values['-LIST_WAV-'][0]
        cmd_audio = AudioSegment.from_wav(selected_path)
        # window['-LIST_WAV-'].update(values['-LIST_WAV-'])
        if cmd_audio is not None:
            window['-TEXT_STATUS-'].update(f"{values['-LIST_WAV-'][0]}, {cmd_audio.duration_seconds}s, {cmd_audio.dBFS:.2f}dBFS")
        else:
            window['-TEXT_STATUS-'].update('')

    if event == '-BTN_PLAY-' and cmd_audio is not None:
        play(cmd_audio)

    if event == '-BTN_DEL-' and cmd_audio is not None:
        os.remove(selected_path)
        cmd_audio = None
        window['-TEXT_STATUS-'].update('')

    if event == '-BTN_VAD-' and cmd_audio is not None:
        # print(selected_path)
        vad_and_save(str(selected_path))

    if event == '-BTN_TRIM-' and cmd_audio is not None:
        trim_and_save(str(selected_path))

    if event == '-BTN_TRIM_MID-' and cmd_audio is not None:
        trim_mid_and_save(str(selected_path))

    try:
        window['-LIST_WAV-'].update([file for file in os.listdir(values['-TEXT_CWD-']) if file.endswith(".wav")])
    except Exception as e:
        print(e)


window.close()
