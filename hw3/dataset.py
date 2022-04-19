




# import jsonlines
# cnt = 1
# with jsonlines.open('./data/train.jsonl', 'r') as f:
#     for data in f:
#         if not cnt:
#             exit()
#         for k, v in data.items():
#             print(k, end = ': ')
#             try:
#                 if len(v) < 50:
#                     print(v)
#                 else:
#                     print(len(v))
#             except:
#                 print(v)
#         cnt -= 1
