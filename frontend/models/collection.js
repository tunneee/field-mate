import mongoose from "mongoose";

const collectionSchema = new mongoose.Schema(
  {
    name: { type: String, required: true },
    numberOfImages: { type: Number, default: 0 },
  },
  { timestamps: true }
);

export const CollectionModel = mongoose.model("Collection", collectionSchema);
