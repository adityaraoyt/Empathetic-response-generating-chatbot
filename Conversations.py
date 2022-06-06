from datetime import datetime
class Log:
    def __init__(self):
        pass

    def saveConversations(self, sessionID, usermessage,botmessage,intent,dbConn):

        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")

        #db = dbConn['Covid-19DB']  # connecting to the database called crawlerD
        mydict = {"sessionID":sessionID,"User Intent" : intent ,"User": usermessage, "Bot": botmessage, "Date": str(self.date) + "/" + str(self.current_time)}

        #table = db[sessionID]
        records = dbConn.chat_records
        records.insert_one(mydict)
