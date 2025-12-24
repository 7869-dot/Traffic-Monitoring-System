"""
Arduino Controller Module
Handles communication with Arduino devices for traffic light control
"""
import serial
import time
from typing import Optional, Dict, Any

class ArduinoController:
    """Controller for Arduino traffic light system"""
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 9600):
        """
        Initialize Arduino controller
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Serial communication baud rate
        """
        self.port = port
        self.baudrate = baudrate
        self.connection: Optional[serial.Serial] = None
        self.last_command: Optional[str] = None
    
    def connect(self, port: Optional[str] = None) -> bool:
        """
        Connect to Arduino
        
        Args:
            port: Serial port (uses self.port if not provided)
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            port = port or self.port
            if port is None:
                # Try to auto-detect port (you may need to implement this)
                return False
            
            self.connection = serial.Serial(port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            return True
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.connection and self.connection.is_open:
            self.connection.close()
            self.connection = None
    
    def is_connected(self) -> bool:
        """Check if Arduino is connected"""
        return self.connection is not None and self.connection.is_open
    
    def send_command(self, action: str, value: Optional[int] = None, duration: Optional[int] = None) -> Dict[str, Any]:
        """
        Send command to Arduino
        
        Args:
            action: Command action (e.g., 'red', 'green', 'yellow', 'blink')
            value: Optional value for the command
            duration: Optional duration in milliseconds
        
        Returns:
            Dictionary with command result
        """
        if not self.is_connected():
            if not self.connect():
                raise ConnectionError("Arduino not connected")
        
        try:
            # Format command string
            command = f"{action}"
            if value is not None:
                command += f":{value}"
            if duration is not None:
                command += f":{duration}"
            command += "\n"
            
            # Send command
            self.connection.write(command.encode())
            self.last_command = action
            
            # Wait for response (optional)
            time.sleep(0.1)
            response = self.connection.readline().decode().strip()
            
            return {
                "success": True,
                "command": action,
                "response": response
            }
        except Exception as e:
            raise Exception(f"Failed to send command: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.disconnect()

