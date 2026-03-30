"""
UART communication module for RBC2026 Robocon Vision System.

This module handles serial communication with robot control system at 50Hz.
"""

import serial
import threading
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Constants
DEFAULT_PORT = '/dev/ttyUSB0'
DEFAULT_BAUDRATE = 115200
DEFAULT_FREQUENCY_HZ = 50
UART_INTERVAL = 1.0 / DEFAULT_FREQUENCY_HZ  # 0.02 seconds for 50Hz
MAX_ERROR_VALUE = 9999


class UARTError(Exception):
    """Custom exception for UART-related errors."""
    pass


class UARTManager:
    """
    Manages UART communication with robot control system.
    
    Sends error values at fixed frequency (50Hz) in background thread.
    Gracefully degrades when UART port is not available.
    """
    
    def __init__(self, port: str = DEFAULT_PORT, baudrate: int = DEFAULT_BAUDRATE):
        """
        Initialize UART manager.
        
        Args:
            port: Serial port path (e.g., '/dev/ttyUSB0')
            baudrate: Baud rate for serial communication
        
        Note:
            If port is not available, system will run without UART (graceful degradation).
        """
        self.port = port
        self.baudrate = baudrate
        self.logger = logging.getLogger(f"{__name__}.UARTManager")
        
        # Thread safety
        self.lock = threading.Lock()
        self.last_packet = "S+0000E\n"
        self.running = True
        self.ser: Optional[serial.Serial] = None
        
        # Connection status
        self.connected = False
        
        # Initialize serial connection
        self._init_serial()
        
        # Start background send thread
        self.thread = threading.Thread(target=self._send_loop_50hz, daemon=True)
        self.thread.start()
    
    def _init_serial(self) -> None:
        """Initialize serial connection."""
        try:
            self.ser = serial.Serial(
                self.port, 
                self.baudrate, 
                timeout=0, 
                write_timeout=None
            )
            self.connected = True
            self.logger.info(f"UART initialized successfully on {self.port} at {self.baudrate} baud")
        
        except serial.SerialException as e:
            self.logger.warning(
                f"UART port not available ({self.port}): {e}. "
                "Running in no-UART mode."
            )
            self.connected = False
            self.ser = None
        
        except Exception as e:
            self.logger.error(f"Unexpected error initializing UART: {e}")
            self.connected = False
            self.ser = None
    
    def _send_loop_50hz(self) -> None:
        """
        Background thread that sends UART packets at 50Hz frequency.
        
        Uses precise timing to maintain consistent send rate.
        """
        interval = UART_INTERVAL
        next_t = time.perf_counter()
        
        while self.running:
            # Only send if serial connection is available and open
            if self.ser and self.ser.is_open:
                try:
                    # Get latest packet thread-safely
                    with self.lock:
                        packet = self.last_packet
                    
                    # Send packet
                    self.ser.write(packet.encode())
                    self.ser.flush()  # Ensure data is transmitted to the hardware
                
                except serial.SerialException as e:
                    self.logger.warning(f"UART write error: {e}. Attempting reconnection...")
                    self.connected = False
                    self._reconnect()
                
                except Exception as e:
                    self.logger.error(f"Unexpected UART error: {e}")
            
            # Precise timing for 50Hz
            next_t += interval
            sleep_time = next_t - time.perf_counter()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If we're behind schedule, reset timing
                next_t = time.perf_counter()
    
    def _reconnect(self) -> None:
        """Attempt to reconnect to UART port."""
        try:
            if self.ser:
                self.ser.close()
            
            time.sleep(0.1)  # Brief delay before reconnection
            
            self.ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=0,
                write_timeout=None
            )
            self.connected = True
            self.logger.info(f"UART reconnected successfully on {self.port}")
        
        except Exception as e:
            self.logger.warning(f"UART reconnection failed: {e}")
            self.connected = False
            self.ser = None
    
    def send_error(self, error_x: int) -> None:
        """
        Update error value to be sent via UART.
        
        Packet format: "S{sign}{value:04d}E\n"
        Example: "S+0123E\n" for error_x=123, "S-0045E\n" for error_x=-45
        
        Args:
            error_x: Error value in pixels (will be clamped to [-9999, 9999])
        """
        try:
            # Clamp error value
            error_clamped = max(-MAX_ERROR_VALUE, min(MAX_ERROR_VALUE, int(error_x)))
            
            # Format packet
            sign = "+" if error_clamped >= 0 else "-"
            value = abs(error_clamped)
            packet = f"S{sign}{value:04d}E\n"
            
            # Update packet thread-safely
            with self.lock:
                self.last_packet = packet
        
        except Exception as e:
            self.logger.error(f"Error formatting UART packet: {e}")
    
    def is_connected(self) -> bool:
        """
        Check if UART connection is active.
        
        Returns:
            True if connected, False otherwise
        """
        return self.connected and self.ser is not None and self.ser.is_open
    
    def stop(self) -> None:
        """Stop UART manager and close connection."""
        self.logger.info("Stopping UART manager...")
        self.running = False
        
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
                self.logger.info("UART connection closed")
            except Exception as e:
                self.logger.error(f"Error closing UART: {e}")
        
        self.connected = False
