import { useEffect, useState } from "react";
import "./App.css";

import SoundCircle from "./components/SoundCircle";

function App() {
  const [query, setQuery] = useState<string>("What is an API?");
  const [response, setResponse] = useState<string>("");
  const [APIRequestCompleted, setAPIRequestCompleted] =
    useState<boolean>(false);

  useEffect(() => {
    queryDracieGPT();
  }, []);

  const queryDracieGPT = async () => {
    try {
      setAPIRequestCompleted(false);
      const response = await fetch(
        "http://localhost:8000/api/query/?query=" + query
      );
      const data = await response.json();
      console.log(data.value);
      setResponse(data.value);
    } catch (err) {
      console.log(err);
    } finally {
      setAPIRequestCompleted(true);
    }
  };

  return (
    <>
      <SoundCircle />
      {APIRequestCompleted ? <p>{response}</p> : null}
    </>
  );
}

export default App;
