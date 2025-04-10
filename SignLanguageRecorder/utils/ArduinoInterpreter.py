import asyncio, struct, time, platform
import numpy as np

from bleak import BleakClient, BleakScanner
from datetime import datetime, timedelta

class ArduinoInterpreter:
    RIGHT_SERVICE_UUID = "fb349b5f-8000-0080-1ef7-000012345678"
    LEFT_SERVICE_UUID = "fb349b5f-8000-0080-417e-000012345678"
    
    CHARACTERISTIC_UUID = "0000fb34-9b5f-8000-0080-001000005678"

    LEFT_IMU_ADDRESS = "CF:A1:D7:91:BE:0D"
    RIGHT_IMU_ADDRESS = "CE:13:DE:BD:F6:C2"

    clients = [None, None]

    timestampStore = 0
    TIME_TO_INIT = 1000  # in ms

    # Calibration Settings for each Arduino
    magnet_calibrations = [[-1512, 1705, -1048, 1395, -862, 2525],[-1617, 1652, -2645, 820, -85, 3352]]

    def __init__(self, isConnected, handData):
        self.isConnected = isConnected
        self.dataStorage = handData
        self.data_queue = asyncio.Queue()

        # Add asyncio.Condition for synchronization
        self.connection_condition = asyncio.Condition()

    async def __sendTimestamp(self, client):
        data = [self.timestampStore, self.timestampStore + self.TIME_TO_INIT]
        packed_data = struct.pack('ii', *data)
        
        try:
            await client.write_gatt_char(self.CHARACTERISTIC_UUID, packed_data)
        except Exception as e:
            raise Exception(f"Code Error: Failed to send timestamp: {e}")

    def __fetch_timestamp(self):
        print("Gloves Connected")
        custom_epoch = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        epoch_ms = int(custom_epoch.timestamp() * 1000)
        self.timestampStore = int(time.time() * 1000) - epoch_ms

    async def __sendCalibration(self, device_ID, client):
        print("Sending Calibration Data")
        data = self.magnet_calibrations[device_ID]
        packed_data = struct.pack('6f', *data)

        try:
            await client.write_gatt_char(self.CHARACTERISTIC_UUID, packed_data)
        except Exception as e:
            raise Exception(f"Code Error: Failed to send Calibration Data: {e}")
        
    async def __awaitConnections(self, device_ID, client):
        async with self.connection_condition:
            self.isConnected[device_ID] = True

            if all(self.isConnected):
                self.__fetch_timestamp()

            self.connection_condition.notify_all()
            # Wait until both devices are connected
            await self.connection_condition.wait_for(lambda: all(self.isConnected))

    async def __bluetoothConn(self, DEVICE_ADDRESS, device_ID):
        while True:
            async with self.connection_condition:
                self.isConnected[device_ID] = False
                self.connection_condition.notify_all()

            client = None

            try:
                device = await BleakScanner.find_device_by_address(DEVICE_ADDRESS, timeout=20.0)

                print("Attempting Connect Device: {}".format(device_ID))
                async with BleakClient(device) as client:
                    print(f"Glove {device_ID} Connected")
                    self.clients[device_ID] = client

                    # await self.__sendCalibration(device_ID, client)
                    await self.__awaitConnections(device_ID, client)

                    def notification_handler(sender, data):
                        if len(data) >= 60:
                            sensor_values = struct.unpack('15f', data)
                            asyncio.create_task(self.data_queue.put(np.hstack([device_ID, sensor_values]).tolist()))
                        else:
                            raise Exception("Code Error: Unexpected data length or format")
                        
                    await client.start_notify(self.CHARACTERISTIC_UUID, notification_handler)
                    await self.__sendTimestamp(client)

                    while True:
                        if not client.is_connected:
                            raise Exception("Device disconnected")
                        await asyncio.sleep(1)

            except Exception as e:
                if "Code Error" in str(e):
                    raise Exception(e)
                
                if client and client.is_connected:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass
                await asyncio.sleep(5)

    async def process_data_queue(self):
        while True:
            item = await self.data_queue.get()
            self.dataStorage.append(item)  # Append to shared handData list

    async def fetchMacOsUUID(self, target_service_uuid):
        for i in range(3):
            devices = await BleakScanner.discover(return_adv=True)

            for device, advertisedData in devices.values():
                service_uuids = advertisedData.service_uuids

                if len(service_uuids) == 1 and service_uuids[0] == target_service_uuid:
                    return device.address
                    
    async def get_system_address_config(self):
        operating_os = platform.system()

        if operating_os == "Darwin":
            print("Searching for adresses")
            LEFT_ADDRESS = await self.fetchMacOsUUID(self.LEFT_SERVICE_UUID)
            RIGHT_ADDRESS = await self.fetchMacOsUUID(self.RIGHT_SERVICE_UUID)

            return LEFT_ADDRESS, RIGHT_ADDRESS
        
        else:
            return self.LEFT_IMU_ADDRESS, self.RIGHT_IMU_ADDRESS

    async def run(self, hands):
        left_address, right_address = await self.get_system_address_config()

        asyncio.create_task(self.process_data_queue())  # Start processing queue

        tasks = []

        if hands[0]:
            tasks.append(asyncio.create_task(self.__bluetoothConn(left_address, 0)))
        if hands[1]:
            tasks.append(asyncio.create_task(self.__bluetoothConn(right_address, 1)))

        print("Connecting Gloves")
        await asyncio.gather(*tasks, return_exceptions=True)