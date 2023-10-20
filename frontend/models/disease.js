import mongoose from "mongoose";

const diseaseSchema = new mongoose.Schema(
  {
    name: { type: String, required: true },
  },
  { timestamps: true }
);

export const DiseaseModel = mongoose.model("Disease", diseaseSchema);
