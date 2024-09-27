if intend_length <= max_length:
        sequence_length = intend_length
        output_tokens = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            top_k = 5,
            max_length=sequence_length,
        )
    else:
        output_tokens = torch.tensor(())
        create_output_token = True
        i = 1
        while intend_length >= max_length:
            gen_tokens = model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=0.9,
                    top_k = 5,
                    max_length=max_length,
                )
            
            if create_output_token == True:
                #first round
                output_tokens = gen_tokens
                create_output_token = False
                intend_length = intend_length - max_length 

            else:


                output_tokens = torch.cat((output_tokens, gen_tokens[0][-i*step_size:].unsqueeze_(dim = 0)), 1)

            input_ids = gen_tokens[0][step_size:].unsqueeze_(dim = 0)
            intend_length = intend_length - step_size
            i = i + 1
