export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  graphql_public: {
    Tables: {
      [_ in never]: never
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      graphql: {
        Args: {
          operationName?: string
          query?: string
          variables?: Json
          extensions?: Json
        }
        Returns: Json
      }
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
  public: {
    Tables: {
      job_anomaly_analysis: {
        Row: {
          analysis_data: Json | null
          job_listing_id: string
          last_analyzed_at: string | null
        }
        Insert: {
          analysis_data?: Json | null
          job_listing_id: string
          last_analyzed_at?: string | null
        }
        Update: {
          analysis_data?: Json | null
          job_listing_id?: string
          last_analyzed_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "job_anomaly_analysis_job_listing_id_fkey"
            columns: ["job_listing_id"]
            isOneToOne: true
            referencedRelation: "job_listings"
            referencedColumns: ["id"]
          },
        ]
      }
      job_listings: {
        Row: {
          application_url: string | null
          company_name: string
          created_at: string | null
          description: string | null
          flexibility: string | null
          id: string
          industry: string
          job_title: string
          level: string | null
          location: string | null
          processed_at: string | null
          processed_for_matching: boolean | null
          reason_for_match: string | null
          salary_range: string | null
          source: string | null
          status: string | null
        }
        Insert: {
          application_url?: string | null
          company_name: string
          created_at?: string | null
          description?: string | null
          flexibility?: string | null
          id: string
          industry: string
          job_title: string
          level?: string | null
          location?: string | null
          processed_at?: string | null
          processed_for_matching?: boolean | null
          reason_for_match?: string | null
          salary_range?: string | null
          source?: string | null
          status?: string | null
        }
        Update: {
          application_url?: string | null
          company_name?: string
          created_at?: string | null
          description?: string | null
          flexibility?: string | null
          id?: string
          industry?: string
          job_title?: string
          level?: string | null
          location?: string | null
          processed_at?: string | null
          processed_for_matching?: boolean | null
          reason_for_match?: string | null
          salary_range?: string | null
          source?: string | null
          status?: string | null
        }
        Relationships: []
      }
      saved_jobs: {
        Row: {
          application_url: string | null
          applied_at: string | null
          company_name: string
          created_at: string | null
          id: number
          industry: string | null
          job_description: string | null
          job_title: string
          location: string | null
          notes: string | null
          original_job_id: string | null
          salary_range: string | null
          status: string | null
          updated_at: string | null
          user_session_key: string
        }
        Insert: {
          application_url?: string | null
          applied_at?: string | null
          company_name: string
          created_at?: string | null
          id?: number
          industry?: string | null
          job_description?: string | null
          job_title: string
          location?: string | null
          notes?: string | null
          original_job_id?: string | null
          salary_range?: string | null
          status?: string | null
          updated_at?: string | null
          user_session_key: string
        }
        Update: {
          application_url?: string | null
          applied_at?: string | null
          company_name?: string
          created_at?: string | null
          id?: number
          industry?: string | null
          job_description?: string | null
          job_title?: string
          location?: string | null
          notes?: string | null
          original_job_id?: string | null
          salary_range?: string | null
          status?: string | null
          updated_at?: string | null
          user_session_key?: string
        }
        Relationships: []
      }
      work_experiences: {
        Row: {
          created_at: string | null
          id: string
          narrative_story: string
          original_input: string
          session_id: string
          story_length: number | null
          structured_story: string
          total_rounds: number | null
          updated_at: string | null
        }
        Insert: {
          created_at?: string | null
          id?: string
          narrative_story: string
          original_input: string
          session_id: string
          story_length?: number | null
          structured_story: string
          total_rounds?: number | null
          updated_at?: string | null
        }
        Update: {
          created_at?: string | null
          id?: string
          narrative_story?: string
          original_input?: string
          session_id?: string
          story_length?: number | null
          structured_story?: string
          total_rounds?: number | null
          updated_at?: string | null
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      delete_old_job_listings: {
        Args: Record<PropertyKey, never>
        Returns: undefined
      }
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DefaultSchema = Database[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof (Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        Database[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? (Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      Database[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof Database },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof Database },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends { schema: keyof Database }
  ? Database[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  graphql_public: {
    Enums: {},
  },
  public: {
    Enums: {},
  },
} as const
