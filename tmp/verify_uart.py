import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import time

# Add current dir to sys.path
sys.path.append(os.getcwd())

from core.connection import UARTManager

class TestUARTProtocol(unittest.TestCase):
    @patch('serial.Serial')
    def test_packet_format(self, mock_serial_class):
        # Setup mock
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_ser.in_waiting = 0
        mock_serial_class.return_value = mock_ser
        
        manager = UARTManager(port="MOCK", baudrate=115200)
        
        try:
            # Test positive error
            manager.send_error(123)
            time.sleep(0.1) # Wait for thread to send
            
            # Check if any call contains S+0123E
            found = False
            for call in mock_ser.write.call_args_list:
                if b"S+0123E\n" == call[0][0]:
                    found = True
                    break
            self.assertTrue(found, "Packet S+0123E\n not found in serial writes")
            
            # Test negative error
            manager.send_error(-45)
            time.sleep(0.1)
            
            found = False
            for call in mock_ser.write.call_args_list:
                if b"S-0045E\n" == call[0][0]:
                    found = True
                    break
            self.assertTrue(found, "Packet S-0045E\n not found in serial writes")
            
            # Test 999 (Lost)
            manager.send_error(999)
            time.sleep(0.1)
            
            found = False
            for call in mock_ser.write.call_args_list:
                if b"S+0999E\n" == call[0][0]:
                    found = True
                    break
            self.assertTrue(found, "Packet S+0999E\n not found in serial writes")
            
            # Test state sync read
            mock_ser.in_waiting = 2
            mock_ser.read.return_value = b"1"
            cmd = manager.get_latest_command()
            self.assertEqual(cmd, "1")
            
        finally:
            manager.stop()

if __name__ == "__main__":
    unittest.main()
