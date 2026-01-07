import torch
import torch.nn.functional as F
import math
from typing import List, Tuple


def generalized_advantage_estimation(rewards: List[float], values: List[float], gamma: float, lam: float) -> Tuple[List[float], List[float]]:
    # based on temporal difference error (TD Error)
    
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        # delta_t = r_t + gamma*V(s_t+1) - V(s_t)
        # here we add to the reward calculated in advance a scaled difference between 
        # the critcs expectation about future prediction and the ground trueth
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    
    return advantages, returns


def get_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def train(
    model, 
    dataloader, 
    optimizer, 
    reward_pipe, 
    tokenizer, 
    device, 
    grad_accum_steps: int = 4, 
    epochs: int = 3, 
    max_grad_norm: float = 0.5, 
    scheduler=None, 
    clip_range: float = 0.2, 
    gamma: float = 0.99, 
    lam: float = 0.95, 
    kl_coef: float = 0.05,
    value_loss_coef: float = 0.5,
    adv_clip_range: float = 10.0,
    log_ratio_clip_range: float = 10.0
) -> float:
    
    # L_total = L_policy + c1*L_value - c2*L_entropy

    for epoch in range(epochs):
        print(f"\n{'='*60}\nEpoch {epoch + 1}/{epochs}\n{'='*60}")

        epoch_loss_sum = 0.0
        epoch_policy_loss_sum = 0.0
        epoch_value_loss_sum = 0.0
        epoch_reward_sum = 0.0
        epoch_kl_sum = 0.0
        steps_count = 0
        optimizer.zero_grad()

        num_batches = len(dataloader)

        for step, batch in enumerate(dataloader, 1):
            prompt_texts = batch 
            
            prompt_inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            prompt_ids = prompt_inputs.input_ids
            prompt_length = prompt_ids.shape[1]

            try:
                print(f"\n[Step {step}/{num_batches}] Generating...", end=" ", flush=True)
                
                model.eval()
                with torch.no_grad():
                    gen_output = model.generate(
                        prompt_ids,
                        attention_mask=prompt_inputs.attention_mask,
                        max_new_tokens=16,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    print("Done", flush=True)

                    print(f"  Rewards...", end=" ", flush=True)
                    gen_text = tokenizer.batch_decode(gen_output, skip_special_tokens=True)

                    rewards = reward_pipe(gen_text)
                    if not isinstance(rewards, list):
                        rewards = [rewards]
                    
                    mean_reward = sum(rewards) / len(rewards)
                    print(f"Done {mean_reward:.4f}", flush=True)

                    attention_mask = (gen_output != tokenizer.pad_token_id).long()
                    shift_labels = gen_output[:, 1:].contiguous()
                    batch_size = gen_output.shape[0]
                    seq_len = gen_output.shape[1]
                    generated_length = seq_len - prompt_length

                    print(f"  Ref model...", end=" ", flush=True)
                    ref_logits, _ = model(gen_output, attention_mask, use_ref_model=True)
                    ref_shift_logits = ref_logits[:, :-1, :].contiguous()
                    ref_log_probs = get_log_probs(ref_shift_logits, shift_labels)
                    print("Done", flush=True)

                    print(f"  Old policy...", end=" ", flush=True)
                    old_logits, old_values = model(gen_output, attention_mask, use_ref_model=False)
                    old_shift_logits = old_logits[:, :-1, :].contiguous()
                    old_log_probs = get_log_probs(old_shift_logits, shift_labels)
                    
                    old_values_clone = old_values.clone()
                    old_log_probs_clone = old_log_probs.clone()
                    print("Done", flush=True)

                    # kl_val = log(pi_theta(a_t))-log(pi_ref(a_t))
                    # in other words here we compute the diff between base_weight and the model that is 
                    # training at the moment t
                    # the scope of this op is to prevent reward hacking
                    kl_per_token = old_log_probs_clone - ref_log_probs
                    mean_kl = kl_per_token.mean().item()
                    
                    all_advantages = []
                    all_returns = []
                    
                    for i in range(batch_size):
                        seq_values_list = old_values_clone[i].cpu().float().tolist()
                        
                        token_rewards = []
                        for t in range(seq_len - 1):
                            kl_penalty = -kl_coef * kl_per_token[i, t].item()
                            token_rewards.append(kl_penalty)
                        
                        # this step is named reward shaping and what it does is that splits the reward 
                        # between all generated tokens
                        generated_start_idx = max(0, seq_len - generated_length - 1)
                        reward_per_token = rewards[i] / max(1, generated_length)
                        for t in range(generated_start_idx, seq_len - 1):
                            token_rewards[t] += reward_per_token
                        
                        # gae measures how good the action we just took was, compared to how good we thought it was
                        advantages, returns = generalized_advantage_estimation(token_rewards, seq_values_list, gamma, lam)
                        
                        all_advantages.append(torch.tensor(advantages, device=device, dtype=torch.float32))
                        all_returns.append(torch.tensor(returns, device=device, dtype=torch.float32))
                    
                    advantages_tensor = torch.stack(all_advantages)
                    returns_tensor = torch.stack(all_returns)
                    
                    adv_mean = advantages_tensor.mean()
                    adv_std = advantages_tensor.std()
                    advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8)
                    advantages_tensor = torch.clamp(advantages_tensor, -adv_clip_range, adv_clip_range)

                model.train()
                print(f"  Training...", end=" ", flush=True)
                
                logits, values = model(gen_output, attention_mask, use_ref_model=False)
                
                shift_logits = logits[:, :-1, :].contiguous()
                new_log_probs = get_log_probs(shift_logits, shift_labels)
                
                log_ratio = new_log_probs - old_log_probs_clone
                log_ratio = torch.clamp(log_ratio, -log_ratio_clip_range, log_ratio_clip_range)
                ratio = torch.exp(log_ratio)
                
                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages_tensor
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values.float(), returns_tensor.float())
                value_loss = torch.clamp(value_loss, 0, 100)
                
                loss = (policy_loss + value_loss_coef * value_loss) / grad_accum_steps

                if not math.isfinite(loss.item()):
                    print(f"\nNon-finite loss: {loss.item()}")
                    optimizer.zero_grad()
                    continue
                
                if not math.isfinite(policy_loss.item()):
                    print(f"\nNon-finite policy loss")
                    optimizer.zero_grad()
                    continue
                    
                if not math.isfinite(value_loss.item()):
                    print(f"\nNon-finite value loss")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                
                if step % grad_accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                    
                    if not math.isfinite(grad_norm):
                        print(f"\nNon-finite gradients")
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()
                    
                    if scheduler is not None:
                        scheduler.step()
                    
                    optimizer.zero_grad()
                    
                    print("Done", flush=True)
                else:
                    grad_norm = 0.0
                    print("Accumulated", flush=True)

                step_loss = float(loss.detach() * grad_accum_steps)
                step_policy_loss = float(policy_loss.detach())
                step_value_loss = float(value_loss.detach())
                
                epoch_loss_sum += step_loss
                epoch_policy_loss_sum += step_policy_loss
                epoch_value_loss_sum += step_value_loss
                epoch_reward_sum += mean_reward
                epoch_kl_sum += mean_kl
                steps_count += 1

                if step % grad_accum_steps == 0:
                    print(f"  Loss: {step_loss:.4f} | Policy: {step_policy_loss:.4f} | Value: {step_value_loss:.4f} | KL: {mean_kl:.4f} | Grad: {grad_norm:.4f}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("\n[WARN] CUDA OOM")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                print(f"\n[ERROR] {str(e)}")
                import traceback
                traceback.print_exc()
                optimizer.zero_grad()
                continue

            if step % 10 == 0:
                avg_loss = epoch_loss_sum / max(1, steps_count)
                avg_reward = epoch_reward_sum / max(1, steps_count)
                avg_kl = epoch_kl_sum / max(1, steps_count)
                print(f"\n{'='*60}")
                print(f"  Step {step}/{num_batches} Summary:")
                print(f"    Avg Loss:   {avg_loss:.4f}")
                print(f"    Avg Reward: {avg_reward:.4f}")
                print(f"    Avg KL:     {avg_kl:.4f}")
                print(f"{'='*60}\n")

        avg_epoch_loss = epoch_loss_sum / max(1, steps_count)
        avg_epoch_policy_loss = epoch_policy_loss_sum / max(1, steps_count)
        avg_epoch_value_loss = epoch_value_loss_sum / max(1, steps_count)
        avg_epoch_reward = epoch_reward_sum / max(1, steps_count)
        avg_epoch_kl = epoch_kl_sum / max(1, steps_count)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} Completed:")
        print(f"  Total Loss:   {avg_epoch_loss:.4f}")
        print(f"  Policy Loss:  {avg_epoch_policy_loss:.4f}")
        print(f"  Value Loss:   {avg_epoch_value_loss:.4f}")
        print(f"  Mean Reward:  {avg_epoch_reward:.4f}")
        print(f"  Mean KL Div:  {avg_epoch_kl:.4f}")
        print(f"{'='*60}\n")

    return avg_epoch_loss