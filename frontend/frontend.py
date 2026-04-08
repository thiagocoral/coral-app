try:
            start_time = time.time()
            response = requests.post(API_URL, json={"user_input": prompt}, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                res_json = response.json()
                full_response = res_json.get("response", "Erro inesperado no formato da resposta.")
                
                if res_json.get("status") == "error":
                    st.warning(full_response)
                    message_placeholder.markdown("⚠️ Erro na IA.")
                else:
                    # Efeito de digitação
                    displayed_text = ""
                    for char in full_response:
                        displayed_text += char
                        message_placeholder.markdown(displayed_text + "▌")
                        time.sleep(0.01)
                    message_placeholder.markdown(full_response)
                    
                    # Métricas na barra lateral
                    st.sidebar.metric("Latência NAI", f"{end_time - start_time:.2f}s")
                    st.sidebar.success("Dados Processados Localmente 🔒")
            else:
                full_response = f"Erro na API: Status {response.status_code}"
                st.error(full_response)
                message_placeholder.markdown("❌ Falha na comunicação.")

        except Exception as e:
            full_response = f"Falha de conexão: {e}"
            st.error(full_response)
            message_placeholder.markdown("🔌 Sem conexão com o servidor.")