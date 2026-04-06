---
description: Deploy code từ laptop sang Jetson Nano qua SSH và quản lý engine TensorRT
---

# Deploy Workflow

Workflow phát triển trên laptop, deploy lên Jetson Nano qua WiFi/Ethernet.

## Cấu hình ban đầu (1 lần duy nhất)

1. Đảm bảo Jetson Nano và laptop cùng mạng WiFi/Ethernet
2. Setup SSH key:
```bash
ssh-keygen -t rsa        # Nếu chưa có key
ssh-copy-id du@<JETSON_IP>
```
3. Cập nhật IP trong `deploy.sh`:
```bash
export JETSON_IP=<IP_CỦA_JETSON>
```

## Deploy code (hàng ngày)

// turbo-all

1. Sync code từ laptop sang Jetson:
```bash
./deploy.sh
```

2. (Lần đầu hoặc khi thay model) Build engine trên Jetson:
```bash
ssh du@$JETSON_IP "cd ~/jetson-vision && ./setup_jetson.sh"
```

3. Chạy chương trình từ xa:
```bash
./remote_run.sh
```

## Cài systemd service (optional, 1 lần)

```bash
ssh du@$JETSON_IP "sudo cp ~/jetson-vision/jetson-vision.service /etc/systemd/system/ && sudo systemctl enable jetson-vision && sudo systemctl start jetson-vision"
```

Sau đó chỉ cần:
```bash
./deploy.sh && ssh du@$JETSON_IP "sudo systemctl restart jetson-vision"
```
