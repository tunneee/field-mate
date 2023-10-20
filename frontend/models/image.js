import mongoose from "mongoose";

const imageSchema = new mongoose.Schema(
  {
    collectionName: { type: String, required: true },
    imageUrl: { type: String, required: true },
    x_center: { type: Number, required: true },
    y_center: { type: Number, required: true },
    width: { type: Number, required: true },
    height: { type: Number, required: true },
    // diseaseId: { type: String },
    // userId: { type: String, required: true },
  },
  { timestamps: true }
);

export const ImageModel = mongoose.model("Image", imageSchema);
