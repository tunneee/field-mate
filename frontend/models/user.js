import mongoose from "mongoose";

const userSchema = new mongoose.Schema(
  {
    phone: { type: String, required: true },
    password: { type: String, required: true },
    numberOfImages: { type: Number, default: 0 },
  },
  { timestamps: true }
);

export const UserModel = mongoose.model("User", userSchema);
