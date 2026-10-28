import av
import numpy as np
import cv2
import time
import datetime
import csv
import os
from fractions import Fraction

class VideoCapture:
    """
    標準Webカメラおよび動画ファイルの取得クラス。
    ヘッダーの回転情報と引数の追加回転角(rot_deg)を合算し、
    __init__ 内でメモリバッファを事前確保（self生成）することで、
    MediaPipe等の認識精度を維持しつつ低遅延で正位置のフレームを返します。
    """
    def __init__(self, source, rot_deg=0):
        self.source = source
        self._start_time = None
        self._current_msec = 0.0
        self._is_opened = False
        self._range_end_msec = None
        self._is_paused = False
        self._last_frame = None
        self._is_internal_update = False
        
        self.rotation = 0
        self.rot_deg = rot_deg
        self._rot_code = None
        self._rot_buffer = None
        self._total_deg = 0

        if isinstance(source, int):
            self.mode = "camera"
            self._cap = cv2.VideoCapture(source)
            self._container = None
            self._is_opened = self._cap.isOpened()
            
            total_deg = (self.rot_deg % 360 + 360) % 360
            self._set_rotation_code(total_deg)
            
            # --- カメラ用回転キャッシュ行列の生成 ---
            if self._rot_code is not None and self._is_opened:
                w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    if self._rot_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                        self._rot_buffer = np.empty((w, h, 3), dtype=np.uint8)
                    else:
                        self._rot_buffer = np.empty((h, w, 3), dtype=np.uint8)
                        
        elif isinstance(source, str):
            self.mode = "file"
            self._container = av.open(source)
            self._stream = self._container.streams.video[0]
            self._decoder = self._container.decode(video=0)
            self._fps = float(self._stream.average_rate)
            self._time_base = self._stream.time_base
            self._last_pts = 0
            self._is_opened = self._container is not None

            # 1. metadata から 'rotate' を取得
            if 'rotate' in self._stream.metadata:
                try:
                    self.rotation = int(self._stream.metadata['rotate'])
                except ValueError:
                    pass

            # 2. metadata で取れなかった場合、side_data から取得（旧PyAV/新PyAV両対応）
            if self.rotation == 0:
                side_data_list = []
                # PyAVのバージョン変化（stream直接 or codec_context）に対応
                if hasattr(self._stream, 'side_data'):
                    try:
                        side_data_list = self._stream.side_data
                    except AttributeError:
                        pass

                for sd in side_data_list:
                    if hasattr(sd, 'rotation'):
                        self.rotation = int(sd.rotation)
                        break
                    elif hasattr(sd, 'type') and str(sd.type) == 'DISPLAYMATRIX':
                        # DISPLAYMATRIXから角度を計算する場合のフォールバック
                        if hasattr(sd, 'rotation'):
                            self.rotation = int(sd.rotation)
                            break

            total_deg = ((self.rotation + self.rot_deg) % 360 + 360) % 360
            self._set_rotation_code(total_deg)

            # --- 動画ファイル用回転キャッシュ行列の生成 ---
            if self._rot_code is not None:
                w = self._stream.width
                h = self._stream.height
                if self._rot_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                    self._rot_buffer = np.empty((w, h, 3), dtype=np.uint8)
                else:
                    self._rot_buffer = np.empty((h, w, 3), dtype=np.uint8)
        else:
            raise ValueError(f"サポートされていないソース形式です: {type(source)}")

    def _set_rotation_code(self, total_deg):
        """合算後の角度から回転コードを設定"""
        self._total_deg = total_deg
        if total_deg == 90:
            self._rot_code = cv2.ROTATE_90_CLOCKWISE
        elif total_deg == 180:
            self._rot_code = cv2.ROTATE_180
        elif total_deg == 270:
            self._rot_code = cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            self._rot_code = None

    def _apply_fast_rotation(self, raw_bgr):
        """事前確保したバッファへ上書き（dst指定）することでCPU負荷を最小化して回転"""
        if self._rot_code is None:
            return raw_bgr

        if self._rot_buffer is None:
            h, w, c = raw_bgr.shape
            if self._rot_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                self._rot_buffer = np.empty((w, h, c), dtype=np.uint8)
            else:
                self._rot_buffer = np.empty((h, w, c), dtype=np.uint8)

        cv2.rotate(raw_bgr, self._rot_code, dst=self._rot_buffer)
        return self._rot_buffer

    def toggle_pause(self):
        self._is_paused = not self._is_paused
        return self._is_paused

    def seek(self, sec):
        if self.mode != "file":
            raise NotImplementedError("シーク機能は動画ファイルモードでのみ利用可能です。")
        
        target_pts = int(sec / self._time_base)
        target_msec = sec * 1000.0
        
        self._container.seek(target_pts, stream=self._stream)
        self._decoder = self._container.decode(video=0)
        
        for frame in self._decoder:
            self._last_pts = frame.pts
            self._current_msec = float(frame.pts * self._time_base * 1000)
            raw_bgr = frame.to_ndarray(format='bgr24')
            self._last_frame = self._apply_fast_rotation(raw_bgr)
            if self._current_msec >= target_msec:
                break

    def set_range(self, start_sec, duration_sec):
        if self.mode != "file":
            raise NotImplementedError("範囲指定読み出しは動画ファイルモードでのみ利用可能です。")
        self.seek(start_sec)
        self._range_end_msec = (start_sec + duration_sec) * 1000.0

    def read(self):
        if self._is_paused:
            return True, self._last_frame

        if self._start_time is None:
            self._start_time = time.perf_counter()

        if self.mode == "camera":
            ret, frame = self._cap.read()
            if ret:
                self._current_msec = (time.perf_counter() - self._start_time) * 1000.0
                self._last_frame = self._apply_fast_rotation(frame)
            return ret, self._last_frame

        try:
            frame = next(self._decoder)
            self._last_pts = frame.pts
            self._current_msec = float(frame.pts * self._time_base * 1000)

            if self._range_end_msec is not None and self._current_msec > self._range_end_msec:
                return False, None

            raw_bgr = frame.to_ndarray(format='bgr24')
            self._last_frame = self._apply_fast_rotation(raw_bgr)
            return True, self._last_frame
        except StopIteration:
            return False, None

    def get(self, propId):
        """回転後の幅・高さを含む各プロパティ値を返します"""
        if propId == cv2.CAP_PROP_POS_MSEC:
            return self._current_msec

        if self.mode == "camera":
            if propId == cv2.CAP_PROP_POS_FRAMES:
                fps = self._cap.get(cv2.CAP_PROP_FPS)
                fps = fps if fps > 0 else 30.0
                return float(round((self._current_msec * fps) / 1000.0))
            
            w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if self._total_deg in (90, 270):
                w, h = h, w
            
            if propId == cv2.CAP_PROP_FRAME_WIDTH:
                return float(w)
            elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(h)
            return self._cap.get(propId)

        calculated_frame = float(round(self._last_pts * self._time_base * self._fps))
        
        w = float(self._stream.width)
        h = float(self._stream.height)
        
        if self._total_deg in (90, 270):
            w, h = h, w

        mapping = {
            cv2.CAP_PROP_FPS: self._fps,
            cv2.CAP_PROP_FRAME_WIDTH: w,
            cv2.CAP_PROP_FRAME_HEIGHT: h,
            cv2.CAP_PROP_FRAME_COUNT: float(self._stream.frames) if self._stream.frames else 0.0,
            cv2.CAP_PROP_POS_FRAMES: calculated_frame
        }
        return mapping.get(propId, 0.0)

    def release(self):
        if self.mode == "camera":
            self._cap.release()
        elif self._container:
            self._container.close()
            self._container = None
        self._is_opened = False

    def isOpened(self):
        return self._is_opened

    def init_seekbar(self, winname):
        if self.mode != "file":
            return

        total_frames = int(self._stream.frames) if self._stream.frames else 100

        def _on_trackbar(pos):
            if self._is_internal_update:
                return
            if self._fps > 0:
                target_sec = pos / self._fps
                current_pos = self.get(cv2.CAP_PROP_POS_FRAMES)
                if abs(pos - current_pos) > 2:
                    self._is_paused = True
                    self.seek(target_sec)

        def _on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.toggle_pause()

        cv2.createTrackbar("Seek", winname, 0, total_frames, _on_trackbar)
        cv2.setMouseCallback(winname, _on_mouse)

    def imshow(self, winname, frame):
        display_frame = frame.copy()
        
        if self._is_paused:
            cv2.putText(display_frame, "PAUSE", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow(winname, display_frame)
        
        if self.mode == "file":
            self._is_internal_update = True
            try:
                current_frame = int(self.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.setTrackbarPos("Seek", winname, current_frame)
            except Exception:
                pass
            finally:
                self._is_internal_update = False

class VideoWriter:
    """
    カメラ・動画ファイル向けのCFR/VFR対応エンコーダー兼ロガー。
    """
    def __init__(self, filename, fps, frame_size, is_vfr=False):
        self.filename = filename
        self.fps = fps
        self.width, self.height = frame_size
        self.is_vfr = is_vfr
        self.start_time = None
        self.frame_count = 0

        self.container = av.open(filename, mode='w')
        self.stream = self.container.add_stream('libx264', rate=int(fps))
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = 'yuv420p'
        self.stream.codec_context.time_base = Fraction(1, 1000)
        self.stream.options = {'crf': '18', 'preset': 'medium'}

        log_path = os.path.splitext(filename)[0] + ".log.csv"
        self.log_file = open(log_path, 'w', newline='')
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(["# Start NTP", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')])
        self.log_writer.writerow(["# Mode", "VFR" if is_vfr else "CFR"])
        self.log_writer.writerow(["frame_idx", "theoretical_msec", "actual_msec", "delay_msec", "applied_pts"])

    def write(self, frame, custom_msec=None):
        if self.start_time is None:
            self.start_time = time.perf_counter()

        actual_msec = (time.perf_counter() - self.start_time) * 1000.0
        theoretical_msec = self.frame_count * (1000.0 / self.fps)
        applied_pts = int(custom_msec) if (self.is_vfr and custom_msec is not None) else int(round(theoretical_msec))

        frame_rgb = frame[:, :, ::-1]
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        av_frame.pts = applied_pts

        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)

        delay = actual_msec - theoretical_msec
        self.log_writer.writerow([self.frame_count, round(theoretical_msec, 2), round(actual_msec, 2), round(delay, 2), applied_pts])
        self.frame_count += 1

    def release(self):
        try:
            if self.stream:
                for packet in self.stream.encode():
                    self.container.mux(packet)
        except Exception:
            pass
        finally:
            if self.container:
                self.container.close()
                self.container = None
        if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
            self.log_file.close()

    def __del__(self):
        self.release()