###############################################################
#
# Start freeling with:
#
# analyze -f es.cfg --flush -m 0x00000004 --output xml --server --port 5005
#
###############################################################

import socket
import sys
from nltk.corpus import stopwords

import xml.etree.ElementTree as ET

es_stopwords = set(stopwords.words('spanish'))


class FreelingClient(object):

    def __init__(self, host, port, encoding='utf-8', timeout=120.0):
        """Initialise the client, set channel to the path and filename where the server's .in and .out pipes are (without extension)"""
        self.encoding = encoding
        self.BUFSIZE = 10240
        self.socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        self.socket.connect( (host,int(port)) )
        self.encoding = encoding
        self.socket.sendall('RESET_STATS\0'.encode(self.encoding))
        res = self.socket.recv(self.BUFSIZE)
        if not res.decode(self.encoding).strip('\0') == 'FL-SERVER-READY':
            raise Exception("Server not ready")

    def get_response(self, text):
        self.socket.sendall((text + "\n\0").encode(self.encoding))

        results = []
        done = False
        data = b""
        while not done:
            buf = self.socket.recv(self.BUFSIZE)
            if buf[-1] == 0:
                data += buf[:-1]
                done = True
            else:
                data += buf
        return data.decode(self.encoding)

    def process(self, text):
        data = self.get_response(text)
        root = ET.fromstring("<root>" + data + "</root>")
        result = []
        all_tokens = []
        next_has_leading_space = True
        for sentence in root:
            sentence_tokens = []
            rebuilt_sentence = ""
            for token in sentence:
                if token.attrib['tag'].startswith("F"):
                    try:
                        if token.attrib['punctenclose'] == 'open':
                            next_has_leading_space = False
                    except:
                        next_has_leading_space = True
                    if not next_has_leading_space:
                        rebuilt_sentence += " " + token.attrib["form"]
                    else:
                        rebuilt_sentence += token.attrib["form"]
                else:
                    if next_has_leading_space:
                        rebuilt_sentence += " " + token.attrib["form"]
                    else:
                        rebuilt_sentence += token.attrib["form"]
                        next_has_leading_space = True

                    if token.attrib['lemma'] not in es_stopwords:
                        sentence_tokens.append(token.attrib['lemma'])
                        all_tokens.append(token.attrib['lemma'])
            result.append((rebuilt_sentence.strip(), sentence_tokens))
        return result, all_tokens

    def sent_process(self, sentence):
        data = self.get_response(sentence)
        root = ET.fromstring("<root>" + data + "</root>")
        sentence_tokens = []
        for sentence in root:
            sentence_tokens = []
            for token in sentence:
                if not token.attrib['tag'].startswith("F") and \
                        not token.attrib['pos']=='date':
                    if token.attrib['lemma'] not in es_stopwords:
                        sentence_tokens.append(token.attrib['lemma'])
        return sentence_tokens
