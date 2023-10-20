import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import mongoose from "mongoose";

import users from "./routers/users.js";
import images from "./routers/images.js";
import diseases from "./routers/diseases.js";
import collections from "./routers/collections.js";


const app = express();
const PORT = process.env.PORT || 5500;

const URI =
  "mongodb+srv://admin:MNUUqvnVNAK2jq2L@field-mate.yrzm45x.mongodb.net/?retryWrites=true&w=majority";

app.use(bodyParser.json({ limit: "30mb" }));
app.use(bodyParser.urlencoded({ extended: true, limit: "30mb" }));
app.use(cors());

app.use('/users', users);
app.use('/images', images);
app.use('/collections', collections);
app.use('/diseases', diseases);

mongoose
  .connect(URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => {
    console.log("connected to Mongoose");
    app.listen(PORT, () => {
      console.log("Server listening on port", PORT);
    });
  })
  .catch((err) => {
    console.log("err", err);
  });