from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM


if __name__ == "__main__":
    name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(name)
    text = "哈囉，你們好，這裡是台灣"
    print(text)
    print("tokenized:", tokenizer(text))

    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model = model.cuda()

    input()
