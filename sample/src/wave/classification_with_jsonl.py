import elekiban

jsonl_path = "dataset/wave/classification/fixed_length/wave.jsonl"
input_pump = elekiban.pipeline.pump.FixedLengthWavePump(data_path=jsonl_path, axes=("a", "b"))
input_pipeline = elekiban.pipeline.pipe.PipeWithPump("wave_input", input_pump)

output_pump = elekiban.pipeline.pump.LabelPump(data_path="dataset/wave/classification/fixed_length/label.csv")
output_pipeline = elekiban.pipeline.pipe.PipeWithPump("label_output", output_pump)

train_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=3)
valid_faucet = elekiban.pipeline.toolbox.SimpleFaucet([input_pipeline], [output_pipeline], batch_size=3)

model = elekiban.model.wave.classification.get_simple_model("wave_input", "label_output", class_num=1, model_scale=1.0, input_channel=2)
output_path = elekiban.executor.train(model, train_faucet, valid_faucet, epochs=30, output_path="output")
print(output_path)
