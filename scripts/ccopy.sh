#!/bin/sh
scp -r $1 pi@192.168.0.17:/home/pi/$1
scp -r $1 pi@192.168.0.16:/home/pi/$1
scp -r $1 pi@192.168.0.12:/home/pi/$1
scp -r $1 pi@192.168.0.13:/home/pi/$1
scp -r $1 pi@192.168.0.8:/home/pi/$1
scp -r $1 pi@192.168.0.9:/home/pi/$1
scp -r $1 pi@192.168.0.5:/home/pi/$1
scp -r $1 pi@192.168.0.4:/home/pi/$1
scp -r $1 pi@192.168.0.2:/home/pi/$1
