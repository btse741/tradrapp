from ib_insync import Future, IB, ContFuture
import ib_insync



ib = IB()
ib.connect('127.0.0.1', 4001, clientId=1)
contract = Future(symbol='ES', lastTradeDateOrContractMonth='202512', exchange='GLOBEX')
details = ib.reqContractDetails(contract)
expirations = sorted({d.contract.lastTradeDateOrContractMonth for d in details})
ib.disconnect()


print(expirations)
