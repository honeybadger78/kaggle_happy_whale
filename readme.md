## Kaggle Happy Whale Competition Summary
🔗 [Competition Link](https://www.kaggle.com/competitions/happy-whale-and-dolphin/overview)

---

### 🏆 Result:🥉 **127/1588** 🥉

---

### 📸 Dataset Preprocessing:

**Approaches:**
1. **Full Body Cropping**: This method captures the entirety of the subject.
2. **Back Fin Cropping**: This technique emphasizes the back fin with an optimal image size of approximately 768x768 pixels.
3. **Back Fin with No Background**: An enhanced version of the back fin cropping, offering a clearer view.

**Key Insights:**
- A combination of both the full body and back fin techniques is essential for precise identification.
- Relying solely on back fin images could overlook other significant body features.
- **Strategy**: Train two distinct models:
  1. One focusing on Full Body Crops
  2. Another emphasizing Back Fin Crops
  
**🌟 Recommendation**: For optimal outcomes, train using both cropping methodologies and employ ensemble techniques during the post-training phase.

---

### 🎚 Class Balancing in MLP Predictions:

To bolster the prediction accuracy of our MLP model, we devised two strategies. The predictions were formatted in a .csv file, with each row presenting 5 potential match candidates ranked by their confidence levels.

**Strategies:**

1. **Reordering based on Repetition and Rarity**: 
   - If the primary candidate (first element) in a row recurs as the top choice in over 10 other rows, and the subsequent candidate hasn't been the top choice in any other row, their positions are interchanged.
     - E.g., `[ind_1, ind_2, ind_3, …]` is modified to `[ind_2, ind_1, ind_3, …]`.

2. **Prioritizing New Individual**: 
   - If the primary candidate in a row is tagged as `new_individual` and the subsequent candidate hasn't been the top choice in any other row, their positions are swapped.
     - E.g., `[new_ind, ind_1, ind_2, …]` is altered to `[ind_1, new_ind, ind_2, …]`.

**Performance Metrics:**
- MLP without class balancing yielded a **0.781 LB score**.
- MLP with class balancing achieved a **0.809 LB score**.

**🌟 Recommendation**: The incorporation of class balancing strategies significantly enhanced the accuracy of the MLP model.

---
