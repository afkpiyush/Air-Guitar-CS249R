"""
Audio playback engine using Pygame.
Provides ultra-low-latency, non-blocking acoustic guitar chord playback,
sound preloading, volume adjustment, muting, and asset integrity verification.
"""

from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import os
import pygame

from src.config import (
    CHORD_AUDIO_DIR,
    PROJECT_ROOT,
    SUPPORTED_CHORDS,
    AUDIO_SAMPLE_RATE,
    AUDIO_BUFFER_SIZE,
    DEFAULT_VOLUME
)


class AudioEngine:
    """Manages guitar chord audio playback with Pygame mixer."""

    def __init__(
        self,
        audio_dir: Optional[Path] = None,
        volume: float = DEFAULT_VOLUME,
        headless: bool = False
    ):
        self.audio_dir = Path(audio_dir) if audio_dir else CHORD_AUDIO_DIR
        # Fallback to audio_files/chord_recordings if sounds/ does not exist
        if not self.audio_dir.exists():
            fallback = PROJECT_ROOT / "audio_files" / "chord_recordings"
            if fallback.exists():
                self.audio_dir = fallback

        self.volume = max(0.0, min(1.0, volume))
        self.is_muted = False
        self.headless = headless
        self.sounds: Dict[str, pygame.mixer.Sound] = {}
        self.is_initialized = False

        if not self.headless:
            self._init_mixer()
            self._load_sounds()

    def _init_mixer(self) -> bool:
        """Initializes Pygame mixer for low latency."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.pre_init(
                    frequency=AUDIO_SAMPLE_RATE,
                    size=-16,
                    channels=2,
                    buffer=AUDIO_BUFFER_SIZE
                )
                pygame.mixer.init()
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"[AudioEngine Warning] Pygame mixer init failed: {e}. Running in mute/headless mode.")
            self.is_initialized = False
            return False

    def _load_sounds(self) -> None:
        """Preloads all supported chord WAV files into RAM."""
        if not self.is_initialized or not self.audio_dir.exists():
            return

        for chord in SUPPORTED_CHORDS:
            file_path = self.audio_dir / f"{chord}.wav"
            if file_path.exists():
                try:
                    snd = pygame.mixer.Sound(str(file_path))
                    snd.set_volume(self.volume)
                    self.sounds[chord] = snd
                except Exception as e:
                    print(f"[AudioEngine Warning] Could not load sound {file_path}: {e}")
            else:
                pass  # Reported in verify_assets()

    def play_chord(self, chord_name: str) -> bool:
        """
        Triggers non-blocking playback of the specified chord.
        
        Args:
            chord_name: Name of chord (e.g. 'c', 'am', 'g')
            
        Returns:
            True if audio was triggered, False otherwise.
        """
        if self.is_muted or not self.is_initialized:
            return False

        chord_key = chord_name.lower().strip()
        sound = self.sounds.get(chord_key)
        if sound is not None:
            sound.play()
            return True
        else:
            print(f"[AudioEngine] Sound for chord '{chord_key}' not loaded or missing from {self.audio_dir}")
            return False

    def set_volume(self, volume: float) -> None:
        """Sets playback volume between 0.0 and 1.0."""
        self.volume = max(0.0, min(1.0, volume))
        for sound in self.sounds.values():
            sound.set_volume(self.volume)

    def toggle_mute(self) -> bool:
        """Toggles audio mute on/off. Returns new mute state."""
        self.is_muted = not self.is_muted
        return self.is_muted

    def verify_assets(self) -> Dict[str, Any]:
        """
        Verifies presence and loading of all required chord audio files.
        
        Returns:
            Dictionary containing verification status, found files, and missing files.
        """
        found = []
        missing = []
        for chord in SUPPORTED_CHORDS:
            path = self.audio_dir / f"{chord}.wav"
            if path.exists():
                found.append(chord)
            else:
                missing.append(chord)
                
        return {
            "all_present": len(missing) == 0,
            "total_supported": len(SUPPORTED_CHORDS),
            "found_count": len(found),
            "missing_count": len(missing),
            "found_chords": found,
            "missing_chords": missing,
            "audio_directory": str(self.audio_dir)
        }

    def stop_all(self) -> None:
        """Stops all currently playing sounds."""
        if self.is_initialized:
            pygame.mixer.stop()

    def cleanup(self) -> None:
        """Releases Pygame mixer audio resources."""
        if self.is_initialized:
            pygame.mixer.quit()
            self.is_initialized = False
